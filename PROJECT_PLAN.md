# PROJECT_PLAN.md — DocsGPT-Agent

> **What** we are building and in **what order**. Read alongside `CLAUDE.md` (how to build) and
> `DECISIONS.md` (why each choice). All facts marked ⚠️ **VERIFY** must be re-checked against the
> provider's live docs before you rely on them — free-tier limits change frequently.

---

## 0. Problem statement (the thing to say in interviews)

Developer documentation for fast-moving frameworks is a poor retrieval target for naive RAG: answers to
real questions ("how do I stream tokens from a LangGraph node while persisting state?", "what changed
about X between v0.2 and v0.6 and how do I migrate?") are **scattered across multiple doc pages, GitHub
discussions, and changelogs**, and are **entity/token-heavy** (exact API names, flags, version numbers)
where pure semantic (dense) search underperforms. This is the setting where **hybrid retrieval +
reranking + a genuine multi-hop agent** measurably beats "chat with your PDF" — and, crucially, where the
improvement is **quantifiable** with real gold data.

**Why this is not a "numbers on a dataset" project:** it is a working, deployed tool answering real
developer questions with citations, evaluated rigorously. The evaluation exists to *defend* the tool, not
to *be* the project.

---

## 1. Corpus decision (VERIFY licensing before scraping)

### Primary recommendation: LangChain + LangGraph docs + their GitHub
- **Why:** You already know this ecosystem deeply (you're building *with* LangGraph), so you can speak to
  the corpus fluently and author good eval questions. It moves fast → natural version-drift → the "what
  changed between versions" multi-hop story is real, not contrived.
- **Sources:** official docs sites (docs pages), the repo's **GitHub Discussions** (these have a native
  *"marked as answer"* feature → your best source of natural relevance labels), plus **closed Issues** with
  a linked resolving comment/PR, plus `CHANGELOG` / release notes.
  > **→ Superseded by D-018.** GitHub Discussions migrated to the **LangChain Forum** (Discourse) in
  > mid-2025 — `langchain-ai/langchain` has only 4 unanswered Announcement discussions, `langgraph` has
  > none. Natural labels actually come from 163 solved Forum topics; closed Issues were never hand-mapped.
- **Size target:** ~3,000–15,000 chunks (a few hundred doc pages + a few thousand discussion/issue threads).
  Comfortably within Qdrant free tier (~1M vectors @ 768-dim; see §3).

### Alternative: Kubernetes docs + GitHub
- **Why consider:** Kubernetes documentation is **licensed CC BY 4.0** (explicitly reusable — the cleanest
  licensing story), very well-structured, and large. Good if you want to avoid any scraping-ToS ambiguity.
- **Tradeoff:** more stable than LangChain, so the version-drift multi-hop angle is weaker; larger corpus to
  wrangle in limited time.

### ⚠️ Licensing / ToS note (do not skip — an interviewer or another AI will ask)
- **Prefer openly-licensed docs.** Kubernetes = CC BY 4.0. Most framework *code* is MIT/Apache, but **docs
  licensing is separate** — check each source's LICENSE / site terms before scraping.
- **GitHub content** is accessible via the REST/GraphQL API under GitHub's terms; authenticated requests are
  rate-limited to **5,000 requests/hour**. Cache aggressively; don't re-scrape on every run.
- **Avoid corpora whose ToS forbid scraping** (e.g., some commercial docs). When in doubt, pick the CC-licensed
  option. Log the corpus + licensing decision in `DECISIONS.md`.

### ⚠️ Gold-label reality check (this is the part that's easy to underestimate)
"Accepted answers = free labels" is **partly** true and needs manual work:
- **GitHub _Discussions_** have "marked as answer" → clean natural labels. Use these first.
- **GitHub _Issues_ do NOT have accepted answers.** For issues, you must hand-map the resolving comment/PR.
- Plan for **~100–150 hand-verified Q→gold-chunk pairs** as the core gold set (built in Layer 3), augmented
  with RAGAS-synthesized questions. Budget real time for curation; do not assume it's automatic.
> **→ Superseded by D-018/D-025.** Labels came from Forum solved-topics, not GitHub. The reality check
> proved even sharper than expected: 114/163 accepted forum answers link no docs page at all (code
> fixes, not doc pointers) — so the gold set rebalanced to a **26-item real answer-link slice** (the
> honest headline) + **100 synthetic** (Groq-8B generated, gold-by-construction) for statistical power,
> reported separately per source, never blended into one number (D-025).

---

## 2. Architecture (in words)

```
User ──► Vercel (React/TS frontend, free Hobby tier)
           │  HTTPS, streaming
           ▼
     Cloud Run (FastAPI + LangGraph agent)   ← always-free tier, scales to zero
           │
           ├─►  LLM Gateway (one wrapper: routing + 429 backoff + cache)
           │        ├─ Groq llama-3.1-8b-instant  ← cheap high-volume nodes (grade, rewrite, decompose) + eval judge
           │        └─ Gemini 2.5/3.x Flash        ← final answer synthesis
           │
           ├─►  Qdrant Cloud (free tier): dense vectors + BM25 sparse + RRF fusion (server-side)
           │      [→ D-027: qdrant-client's FusionQuery has no k, so fusion is CLIENT-side to honor k=60]
           │
           ├─►  BGE cross-encoder reranker  ← runs in-container on CPU
           │      [→ D-030/D-031/D-032: measured NEGATIVE (worse recall + 14x latency); rejected as
           │       default, kept as a selectable --pipeline rerank ablation]
           │
           └─►  Langfuse Cloud (free tier): every node emits a trace span (tokens, latency, cost)

  Batch jobs (Cloud Run Jobs, on a schedule):
     • nightly re-scrape + re-index corpus
     • eval run (retrieval metrics + RAGAS) → writes results JSON → shown on a dashboard page
```

**Agent flow (the core differentiator, Layer 5):**
`decompose query → (retrieve → grade relevance → rewrite query if weak)×N → synthesize cited answer →
verify answer is grounded in citations → self-correct/retry if not.`
> **→ Superseded by D-037/D-038/D-041.** The built flow is `retrieve → synthesize → verify → {cite |
> retry→synthesize | refuse→cite}`. Query decomposition/multi-query retrieval was built, measured, and
> **rejected** (crushes the real-question slice, D-037). Pre-synthesis relevance grading was built,
> measured, and **rejected** (prior-contaminated — the grader can't separate "I know this" from "the
> passages say this," D-038); replaced by **post-synthesis grounding verification**. Self-correction
> **re-synthesizes over the same context** with corrective feedback, not re-retrieval (D-041) — re-query
> was already shown to hurt, so a retry that re-retrieves would repeat a known failure.

---

## 3. Tech stack (VERIFIED as of the plan date — re-check ⚠️ items)

### Vector store + hybrid retrieval — **Qdrant Cloud free tier**
- **Specs (⚠️ VERIFY):** 0.5 vCPU, 1 GB RAM, 4 GB disk, no credit card, permanent. Holds ~1M vectors @ 768-dim.
- **Why:** stores **dense + BM25 sparse vectors in one collection** and does **RRF fusion server-side**
  (`FusionQuery(fusion=Fusion.RRF)`), so you don't run a separate Elasticsearch. Highest-leverage infra choice.
  > **→ Superseded by D-027.** `qdrant-client` 1.18's `FusionQuery` exposes no `k` parameter, so the
  > documented `k=60` (D-006) couldn't be honored server-side. Fusion runs **client-side** in Python
  > instead — two named-vector queries, fused with `rrf_fuse()`, keeping `k` a real, sweepable tunable.
- **⚠️ CRITICAL OPERATIONAL RISK:** free clusters **auto-suspend after ~1 week of inactivity and are deleted
  after ~4 weeks of inactivity.** Mitigation (build in Layer 8): a one-command **re-index script** so the
  corpus can be rebuilt in minutes, plus a lightweight scheduled **keep-alive ping**. **Never treat the free
  cluster as durable storage for your only copy** — keep the chunked corpus (JSONL) in the repo/bucket.
- **Alternative to mention in interviews:** pgvector on Cloud SQL (the "one database: SQL + vectors + full-text"
  story) — but Cloud SQL is not free beyond the trial, so it spends credits. Keep as the "production alternative I evaluated."

### Embeddings — **BAAI BGE, self-hosted (free)**
- `bge-small-en-v1.5` (**384-dim**, fast on CPU) as default; `bge-base-en-v1.5` (**768-dim**, higher quality) if
  quality needs it. Both fit the free tier easily.
- **Why:** free forever, runs on CPU, no per-call cost across the thousands of embed calls during indexing + eval.
- **Ablation to run (and log):** benchmark `bge-small` vs `bge-base` (and optionally Gemini embeddings) on *your*
  gold set and pick the winner. MTEB/BEIR rankings don't predict your corpus — say this in interviews.
- Note on Gemini embeddings: usable and zero local compute, but **free-tier Gemini may train on your inputs** —
  fine for *public* docs, not for anything private. Log this if you use it.

### Keyword / sparse retrieval + fusion — **Qdrant native BM25 sparse + RRF**
- Standard **RRF**: `score(d) = Σ 1/(k + rank_i(d))`, `k = 60`. Server-side in Qdrant.
  > **→ Superseded by D-027**: fusion is client-side (see above) so `k=60` is actually honored/tunable.
- **Alternative to mention:** `rank_bm25` in-memory (simplest, fine < ~50k chunks) or Postgres full-text.

### Reranker — **BGE cross-encoder, self-hosted (free)**
- `bge-reranker-base` or `bge-reranker-v2-m3` (Apache-2.0). For tighter CPU latency, `ms-marco-MiniLM-L-6-v2`
  (~60–120 ms for ~50 candidates on CPU) or FlashRank.
- Runs in the Cloud Run container on CPU. Measure and report rerank latency.
- **Avoid** depending on Cohere Rerank for the live app — its free/trial quota (⚠️ VERIFY, ~1,000 calls/month) is
  too tight for a public demo. Optionally use it only for a one-off *API-vs-open* comparison ablation.
> **→ Superseded by D-030/D-031/D-032.** bge-reranker-base measured ~25-30s/query on free-tier CPU
> (undeployable); swapped to MiniLM-L6, which measurably **hurt** recall/nDCG and cost 14x latency for
> no gain. A 4-model bake-off (incl. bge via ONNX) confirmed **none** beat hybrid-no-rerank, especially
> on the real forum slice. Reranking is rejected as the default; hybrid alone ships. Kept as a documented,
> selectable ablation (`--pipeline rerank`) because it measurably helps the *synthetic* slice.

### LLMs — **Groq (workhorse) + Gemini (synthesis)**, routed
- **Groq free tier (⚠️ VERIFY):** 30 RPM, 6,000 TPM, and **RPD varies by model** —
  `llama-3.1-8b-instant` ≈ **14,400 RPD** (the workhorse), `llama-3.3-70b-versatile` ≈ **1,000 RPD**.
  No credit card. Open-source models only. Blazing fast.
- **Gemini free tier (⚠️ VERIFY — this shifted recently):** **Pro models are now paid-only.** Only **Flash /
  Flash-Lite** (2.5 and newer 3.x) are free. Published RPD figures vary widely by source and change often
  (seen anywhere from 250 to 1,500 RPD for Flash). **Do not hard-code a number — read the live cap in
  Google AI Studio for your project and design backoff regardless.** Free tier may train on your inputs.
- **Routing strategy (this is a resume-worthy engineering point — log it):**
  - Cheap, high-volume nodes (**query decomposition, relevance grading, query rewriting**) → **Groq 8B**
    (14,400 RPD is huge headroom, and it's fast). This is what keeps you inside free limits.
  - **Final answer synthesis** (quality matters most) → **Gemini Flash** (or Groq 70B as fallback).
  - **Eval-harness LLM-judge calls (RAGAS)** → **Groq 8B** to avoid burning Gemini's small daily quota.
  - All calls through one wrapper with **exponential backoff + jitter** and a **response cache**.
- **Reality to design for:** an agentic query can make 5–15 LLM calls. Routing the cheap ones to Groq 8B is
  what makes a live demo survive on free tier. Say this explicitly in interviews — it shows production maturity.

### Agent framework — **LangGraph**
- **Why:** the cyclic *decompose → retrieve → grade → rewrite → generate → verify* loop is a first-class state
  machine with native cycles, state persistence, and streaming; it renders cleanly in Langfuse traces; and you
  already know LangChain so ramp-up is short. This is what makes the "agentic" claim real rather than decorative.
- **Alternative to mention:** LlamaIndex Workflows/agents (stronger out-of-the-box RAG modules). Reasonable to
  use LlamaIndex retrieval components under LangGraph orchestration.

### Evaluation — **custom retrieval metrics + RAGAS**
- **Retrieval (no LLM judge, fully defensible):** Recall@k (k=1,5,10), MRR, nDCG@10, Hit Rate@k against the gold set.
- **Answer quality (RAGAS):** Faithfulness, Answer Relevancy, Context Precision, Context Recall; Answer Correctness
  where ground-truth answers exist. **RAGAS results depend on the judge LLM — fix the judge model across all runs**
  (use Groq 8B consistently) so ablations are comparable; record the judge in each results file.
- **Gold set:** natural labels (Discussions "marked as answer", curated closed-issue resolutions) as primary +
  RAGAS `TestsetGenerator` synthetic (single-hop + multi-hop) as augmentation; hand-verify ~100–150 items.
  > **→ Superseded by D-018/D-025.** Natural labels = Forum solved topics (26 real answer-link items,
  > the honest headline), not Discussions/Issues. Synthetic = a **self-rolled** Groq-8B generator (not
  > RAGAS `TestsetGenerator`) so gold chunk-ids are known by construction and hop-count is controlled;
  > RAGAS itself is reserved for its real strength — answer-quality judging (Layer 5d, D-042).
- **Build this EARLY (Layer 3), before hybrid/rerank/agent**, so every later change is a tracked delta.

### Observability — **Langfuse Cloud free tier**
- **⚠️ VERIFY** (~50k observations/month, ~30-day retention). Traces nested agent spans (LLM/tool/retrieval),
  token cost, latency; has a scores/eval data model and LangGraph integration.
- **Alternative:** Arize Phoenix (open-source, strong offline RAG-eval visualization). Both are OpenTelemetry-based.

### Deployment
- **Frontend:** React/TS on **Vercel Hobby (free)**. ⚠️ Hobby is **non-commercial** — fine for a portfolio; say so if asked.
- **Backend + agent:** **FastAPI in a container on Google Cloud Run** — always-free tier (⚠️ VERIFY: ~180k vCPU-seconds,
  ~360k GiB-seconds, ~2M requests/month, in specific regions like `us-central1`), **scales to zero**. Reranker on CPU is fine.
  Note: scale-to-zero → **cold starts**; mitigate for demos by warming or a min-instance (spends a little credit).
- **Vector DB:** Qdrant Cloud (managed, not localhost) — satisfies "deployed vector DB".
- **Secrets:** Cloud Run env / Secret Manager. **GCP credits = insurance** (warm instance, pgvector alternative, or a
  one-off GPU batch to precompute embeddings) — not the default.

---

## 4. Build Layers (build ONE at a time — see CLAUDE.md Prime Directive)

Each layer has an **Acceptance Gate**. Do not proceed to the next layer until its gate passes and I approve.
Order is leverage-first: a demoable, *measured* core early; polish and product skin last.

### Layer 0 — Repo scaffold & config
- Repo structure (`src/{ingest,retrieval,agent,eval,api}`, `tests/`, `config.py`/`config.yaml`, `.env.example`,
  `pyproject.toml`/pinned `requirements.txt`, `Makefile`, CI stub, `.gitignore` for `.env`).
- **Gate:** `make setup` installs cleanly; config loads; `.env.example` lists all needed keys; empty smoke test passes.

### Layer 1 — Corpus ingestion & chunking
- Fetch docs (respecting licensing) + GitHub Discussions/Issues (authenticated, cached). Clean HTML/markdown.
- **Structure-aware chunking**: never split a code block or a function signature; keep heading path + source URL +
  version as metadata on every chunk. Write chunks to versioned JSONL (this is your durable copy).
- **Gate:** N chunks produced (report N); each has `{id, text, source_url, heading_path, version, type}`; spot-check
  5 chunks show clean, coherent text with intact code blocks.
  > **→ Superseded by D-016/D-018.** Corpus = a direct `git clone` of `langchain-ai/docs` (.mdx source,
  > SHA-pinned), not HTML scraping — cleaner and licensing-verified MIT. Labels (Layer 1b) come from the
  > Forum's public no-auth JSON API, not authenticated GitHub Discussions/Issues. Result: 11,035 chunks /
  > 751 files.

### Layer 2 — Indexing & dense baseline retrieval
- Embed chunks (bge-small default) → push to Qdrant Cloud (dense only for now) → top-k dense search function.
- **Gate:** a query returns sensible top-k chunks from the **deployed** Qdrant cluster (not local); latency logged.

### Layer 3 — Eval harness + gold set + BASELINE numbers  ← the most important early layer
- Build the gold set (natural labels + RAGAS synthetic; hand-verify ~100–150). Implement retrieval metrics
  (Recall@k, MRR, nDCG, Hit Rate) + wire RAGAS (fixed Groq-8B judge). Run against the **dense-only** baseline.
- **Gate:** baseline numbers printed **and saved to a results file** with the exact config + judge model recorded.
  This is the reference every later improvement is measured against.

### Layer 4 — Hybrid retrieval + reranker (the "money" ablation)
- Add BM25 sparse vectors + RRF fusion in Qdrant. Add BGE reranker over fused candidates. Re-run the full eval.
- **Gate:** a documented before/after table — **dense-only vs hybrid vs hybrid+rerank** — on the same gold set,
  same config. This table alone is a resume bullet.
  > **→ Superseded by D-030/D-031.** The 3-way table was built exactly as planned (`results/eval_{dense,
  > hybrid,rerank}_*.json`) — but the **verdict** flipped from "ship the strongest of the three" to
  > "ship hybrid; reranker is a measured negative, kept only as an ablation." The table is still the
  > resume bullet — it's just an honest rejection, not a win, for the rerank column.

### Layer 5 — Agentic loop + citations
- LangGraph graph: decompose → iterative retrieve → grade → rewrite-on-failure → synthesize → **verify grounding**
  → self-correct/retry. Attach **per-claim citations** to sources. Route nodes per §3 (Groq 8B cheap, Gemini synth).
- **Gate:** multi-hop questions that single-pass RAG failed now succeed; show a Langfuse/graph trace of the loop;
  citations resolve to real chunks.
  > **→ Superseded by D-037/D-038/D-041** (same as the §2 agent-flow note above). Built flow:
  > `retrieve → synthesize → verify(post-hoc grounding) → {cite | retry→synthesize | refuse→cite}`.
  > Decompose/rewrite-on-failure was measured and dropped; "grade" became post-synthesis verification;
  > citations resolve + are validated (`invalid_citations` surfaced, not silently dropped, D-034).

### Layer 6 — API + streaming + rate-limit hardening
- FastAPI endpoints; stream cited answers; the single LLM wrapper with backoff + routing + cache. Measure LLM
  calls/query, tokens/query, p50/p95 latency.
- **Gate:** local API serves a cited, streamed answer end-to-end; backoff demonstrably handles a simulated 429.

### Layer 7 — Observability
- Langfuse spans on every node; cost/latency dashboards; a dashboard/route showing eval history + per-query cost/latency.
- **Gate:** live traces visible in Langfuse; cost & p95 latency per query readable.
  > **→ Implemented per D-045.** Node spans come from a LangChain `CallbackHandler` threaded into the graph;
  > **plus** manual `generation` spans emitted by the LLM gateway (it calls the Groq/Gemini SDKs directly, not
  > LangChain LLMs — D-033 — so the callback can't see the LLM calls / their tokens). Tracing is **optional/
  > no-op** when keys are absent (tests/CLI/eval need no account) and **flushes per request** (Cloud Run
  > scale-to-zero). **No custom in-app dashboard route was built** — cost & **p50/p95 latency** are read from
  > **Langfuse's own trace-aggregation UI** (it provides the percentile dashboard), which satisfies the gate
  > without duplicating a metrics UI. Retrieval/RAGAS eval history remains in the committed `results/*.json`
  > (Layers 3–5d), not folded into the Langfuse view. `langfuse==4.13.2` (OpenTelemetry-based).

### Layer 8 — Deployment (not localhost)
- Containerize; deploy FastAPI to Cloud Run; React to Vercel; wire Qdrant Cloud + Langfuse + keys via Secret Manager.
- Add the **re-index script** and **keep-alive** (Qdrant inactivity mitigation) and a nightly re-scrape/re-index Job.
- **Gate:** a **public URL** answers a question with citations end-to-end; re-index script rebuilds the collection from JSONL.

### Layer 9 (optional, only if time remains) — Product skin
- Let a user point the agent at their **own docs URL / GitHub repo**; saved query history; thumbs-up/down feedback
  that appends to the eval set. Keep per-user corpora isolated.
- **Gate:** a new user can ingest their own docs and query them; feedback is stored and feeds an eval run.

### Layer 10 — Polish & defense
- Final full eval for headline numbers; `README.md` as a "paper" (architecture diagram, ablation tables, honest
  caveats); short demo video; finalized resume bullets backed by reproducible numbers.
- **Gate:** numbers in the README reproduce when the eval is re-run; resume bullets trace to real results files.

**If time runs short:** cut Layer 9 first, then trim Layer 7 to basic tracing. A deployed Layer-0–8 core with the
Layer-4 ablation table and Layer-5 agent is already a standout. **Never cut Layer 3** — the numbers are the point.

---

## 5. Roadmap (maps layers → 5–6 weeks; leverage-first)

| Week | Layers | Outcome |
|---|---|---|
| 1 | 0, 1, 2, **3** | Ingest a slice, dense baseline live on Qdrant, **eval harness + baseline numbers**. |
| 2 | 4 | Hybrid + rerank; **dense vs hybrid vs hybrid+rerank ablation table**. (Highest-ROI deliverable.) |
| 3 | 5 | Agentic loop + citations; multi-hop questions now pass; traces to show. |
| 4 | 6, 7 | API + streaming + backoff/routing; Langfuse observability; cost/latency measured. |
| 5 | 8 (+9 if ahead) | **Deployed public URL**; re-index + keep-alive; optional product skin. |
| 6 | 10 | Final eval, README-as-paper, demo video, resume bullets. |

---

## 6. Metrics to track and put on the resume (frame as ablations, not absolutes)

**Retrieval (compute yourself vs gold — most defensible, no LLM judge):**
- **Recall@5** (headline), Recall@1/@10, **MRR@3**, nDCG@10, Hit Rate@k.

**Answer quality (RAGAS, judge fixed to Groq 8B):**
- Faithfulness (hallucination proxy), Answer Relevancy, Context Precision, Context Recall, Answer Correctness
  (where ground truth exists).

**Hallucination rate (measure honestly):**
- Claim-level: a factual claim contradicted by ground truth (wrong version, invented API, phantom citation).
  **Refusals / "I don't know" are NOT hallucinations.** Report as a before/after of "unsupported-claim rate."
- Read faithfulness *together with* answer relevancy — a near-1.0 faithfulness with low relevancy is a **measurement
  artifact** (little context → nothing to contradict), not a win. Mention this nuance in interviews; it signals rigor.

**Engineering (from Langfuse):**
- p50/p95 end-to-end latency, LLM calls/query, tokens/query, reranker latency, cost/query (notional pay-as-you-go, since you're on free tier).

**Framing (fill X/Y with YOUR measured numbers — never invent):**
- "Hybrid + cross-encoder rerank raised **Recall@5 from X→Y** and **MRR@3 from X→Y** vs a dense-only baseline on a
  150-question gold set." *(Published precedent for the pattern — validate on your own data, cite as precedent only:
  T²-RAGBench reported dense→hybrid+rerank Recall@5 0.587→0.816 and MRR@3 0.433→0.605.)*
- "Query decomposition + self-correction loop **improved multi-hop answer correctness by X%** and **cut
  unsupported-claim rate from X%→Y%** (RAGAS faithfulness 0.9X)."
- "Instrumented Langfuse + an automated RAGAS/recall@k eval harness in CI; tracked **p95 latency, LLM-calls/query,
  cost/query**, and routed cheap agent nodes to Groq to stay within free-tier limits."

**Honesty guardrails:** report *your* measured numbers, keep the RAGAS judge fixed, prefer natural gold labels over
synthetic, and don't claim zero hallucination — production RAG still errs; the point is you **measure and reduce** it.

---

## 7. Open items to re-verify before/while building (⚠️)
- Exact current **Gemini** free-tier RPM/RPD for the specific Flash model you pick (read live in AI Studio).
- Exact current **Groq** per-model RPD (confirm `llama-3.1-8b-instant` daily cap).
- **Qdrant** free-tier inactivity suspend/delete windows (confirm current values; keep the JSONL corpus as backup regardless).
- **Cloud Run** always-free quotas + eligible regions; **Vercel Hobby** limits; **Langfuse** free observation cap.
- Corpus **licensing/ToS** for whichever docs you scrape.
