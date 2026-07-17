# ProjectInfo.md — DocsGPT-Agent Fact Sheet

> **Purpose:** raw material for resume bullets. Feed this file to an LLM *along with a job
> description*; it selects and rephrases the facts below to match that role. Nothing here is
> pre-styled — it's a fact bank, deliberately over-complete.
>
> **Every number below traces to a `results/*.json` file, `README.md`, or a `DECISIONS.md` entry
> (52 entries).** Do not invent metrics beyond these. When quoting a metric, carry its caveat.

---

## 1. One-liners (raw framings, pick per JD)

- **Neutral:** A deployed, agentic, cited question-answering system over LangChain/LangGraph developer
  docs (11,035 chunks / 751 pages) with hybrid retrieval, LangGraph self-correction, per-claim
  citations, a quantitative eval harness, and tracing — entirely on free-tier infrastructure.
- **SDE-leaning:** Designed and shipped a production RAG service to a public URL — FastAPI on Cloud Run
  (scale-to-zero), React/TS on Vercel, SSE streaming, a rate-limit-hardened LLM gateway, Langfuse
  observability, and a one-command disaster-recovery reindex path.
- **DS/ML-leaning:** Built a gold-labeled evaluation harness (126 questions: 26 natural + 100 synthetic)
  and ran a 4-pipeline retrieval ablation that surfaced a synthetic-vs-real evaluation bias, leading to
  two standard "improvements" (cross-encoder reranking, query decomposition) being measured and rejected.
- **Safety/ML-eng-leaning:** Built an agent that structurally refuses rather than hallucinates —
  post-synthesis grounding verification with a bounded self-correction retry, measured against RAGAS.

**Live demo:** https://frontend-three-gamma-49.vercel.app
**API:** https://docsgpt-agent-ee54gmitka-uc.a.run.app (`/docs` OpenAPI, `/ping`, `/ask`, `/ask/stream`; bare `/` 404s by design)
**Repo:** https://github.com/mahendra-kausik/docsGPT

---

## 2. Problem framing (the "why is this hard" angle)

Developer docs for a fast-moving framework are a poor naive-RAG target. Real user questions
("how do I stream tokens from a chain?") are entity/token-heavy (exact API names, args) but are
answered **semantically**, not lexically. Synthetic/generated eval questions, by contrast, are generated
*from* their gold chunk — so they share surface vocabulary with the answer and reward lexical tricks.

**The central empirical finding:** several standard "retrieval sophistication" upgrades measurably
**helped synthetic questions and hurt real ones**. An aggregate-only metric would have hidden this and
shipped a regression against real users. The project's defensibility comes from measuring that gap
per-slice and reporting it honestly.

---

## 3. Headline metrics

### Corpus & data
| Fact | Value |
|---|---|
| Corpus | 11,035 chunks / 751 files, MIT-licensed `langchain-ai/docs` @ sha 662d399 |
| Chunking | fence-aware MDX cleaning + structure-aware chunking (D-017) |
| Gold set | 126 questions = 26 real (LangChain Forum accepted-answer → docs-page, hand-verified) + 100 synthetic (Groq-8B generated, gold-by-construction) |
| Source of truth | `data/corpus/chunks.jsonl` (committed; never re-scraped) |

### Retrieval ablation (4 pipelines, per-slice — the headline table)
| pipeline | overall recall@5 | real recall@5 (n=26) | synthetic recall@5 (n=100) | p50 latency |
|---|---|---|---|---|
| dense (baseline) | 0.612 | **0.577** | 0.622 | 474 ms |
| **hybrid (BM25+RRF) — shipped default** | **0.693** | 0.404 | **0.768** | 523 ms |
| + cross-encoder rerank | 0.638 | 0.327 | 0.718 | 7,306 ms (**14×**) |
| + query decomposition | 0.615 | 0.173 | 0.730 | 2,415 ms (**5×**) |

Shipped hybrid: **recall@5 0.693 / mrr@3 0.616 / ndcg@10 0.634**; synthetic slice recall@5 **0.768
(+24% vs dense)**. Live latency **p50 579 ms / p95 692 ms** end-to-end retrieval; full agent turn
(retrieve + synthesize + verify) **2–4 s warm** on Cloud Run.

### Answer quality (RAGAS)
Judge fixed to Groq `llama-3.1-8b-instant`, routed through the project's own gateway. n=12
(6 forum + 6 synthetic, seed=13) — small by design to stay inside Groq's free-tier TPM cap.
**Directional, not statistically tight.**

| metric | deployed default (Groq 70B synthesis) | earlier run (Groq 8B synthesis) |
|---|---|---|
| faithfulness | 0.361 (n=11) | 0.311 (n=9) |
| answer_relevancy | 0.228 (n=12) | 0.380 (n=12) |
| answer_correctness | 0.381 (n=4) | 0.404 (n=4) |
| context_recall | 0.449 (n=6) | 0.449 (n=6) — identical (retrieval-only metric) |
| context_precision | 0.333 (n=6) | 0.333 (n=6) — identical (retrieval-only metric) |

`context_recall`/`context_precision` are byte-identical across runs because they depend only on
retrieved context, not the synthesis model — a built-in internal-consistency check that the harness
measures what it claims to.

**Absolute numbers are low on purpose and reported as such.** Two known depressants: (1) RAGAS judge
context capped at 4 chunks × 600 chars to stay under Groq's 6,000 TPM free cap, biasing faithfulness
down vs. the 8-chunk context the agent actually sees; (2) free-tier judge calls occasionally fail
(413/parse) — dropped from the mean rather than scored zero, so `n` is reported per-metric.

---

## 4. Architecture / system design

```
User ──HTTPS/SSE──► Vercel (React + TS, Vite)          frontend/
                       ▼
             Cloud Run (FastAPI, scale-to-zero)         src/api/
                       ▼
          LangGraph agent  retrieve → synthesize → verify
                              │            │          │
                              │            │          └─ ungrounded? ─► retry (bounded) ─► synthesize
                              │            │                        └─► refuse ─► cite
                              │            ▼
                              │       LLM Gateway (one wrapper: 429 backoff+jitter, cache, routing)
                              │        ├─ Groq llama-3.3-70b-versatile  (deployed default synthesis)
                              │        ├─ Groq llama-3.1-8b-instant     (verify / cheap nodes / RAGAS judge)
                              │        └─ Gemini 2.5 Flash              (opt-in per request, 20 req/day cap)
                              ▼
                     Qdrant Cloud (free tier)
                       dense (bge-small-en-v1.5, 384-dim, cosine) + BM25 sparse, client-side RRF k=60
                       └─ docs_hybrid (default) / docs_dense (baseline, kept for ablation)

  Langfuse Cloud: every LangGraph node = trace span; every real LLM call = generation span
                  (tokens, latency); optional/no-op with no API key so tests/CLI stay hermetic.
```

**Agent flow (the differentiator):** `retrieve → synthesize → verify(grounding) → {cite | retry→resynthesize | refuse→cite}`

**Repo layout (modular, no monolith):**
```
src/ingest/     corpus fetch + structure-aware chunking
src/retrieval/  embedding, dense/hybrid/rerank retrievers, RRF fusion
src/agent/      LangGraph state machine: retrieve/synthesize/verify/retry/refuse, citations
src/eval/       gold-set build, retrieval metrics, RAGAS harness
src/api/        FastAPI app (JSON + SSE)
src/llm/        provider-aware LLM gateway (routing, backoff, cache)
src/obs/        Langfuse tracing wrapper (optional/no-op)
frontend/       React/TS/Vite UI
results/        every eval run's JSON (config + git SHA + judge recorded)
```

---

## 5. Engineering highlights (SDE-facing)

- **Deployment:** FastAPI + LangGraph agent on Cloud Run (scale-to-zero, `--allow-unauthenticated`);
  React/TS (Vite) on Vercel. **Both models baked into the Docker image** — fixed a cold-start failure
  where HF downloads exceeded the request timeout (D-047). CPU-only torch build to keep the image lean.
- **Streaming:** SSE via `fetch()` + `ReadableStream` (not `EventSource` — the endpoint is POST).
  Streams lifecycle stage events (retrieve/synthesize/verify/retry/refuse), then the **verified**
  answer token-by-token — never the raw pre-verification draft, since verification can retract it.
- **Rate-limit hardening:** a single LLM gateway wraps every Groq/Gemini call with exponential
  backoff + jitter and a response cache; a **simulated-429 recovery path is unit-tested**. No unbounded
  parallel LLM calls.
- **Durability / DR:** the Qdrant free cluster auto-deletes after ~4 weeks idle. A one-command
  `reindex` task rebuilds both collections from the committed `chunks.jsonl`; a **weekly Cloud Scheduler
  job pings `/ping`** to keep the cluster alive (D-048). Recovery procedure verified end-to-end with
  measured timings.
- **Observability:** every LangGraph node and every real LLM call emits a Langfuse span (tokens,
  latency); optional/no-op without an API key so tests, CLI, and eval runs need no account.
- **Security/config:** CORS locked to an explicit origin allow-list; secrets via `.env` locally +
  env vars/Secret Manager in deploy, never in git; all tunables in one config (chunk size, top-k,
  RRF k, rerank top-n, model names) — which is what made the ablations trivial to run.
- **CI/quality:** pinned dependency versions, ruff linting, ~80+ tests, smoke test per layer.
- **Post-deploy debugging:** fixed Gemini fabricating code args via prompt hardening (D-050); fixed
  citation 404s caused by a missing `/python/` URL language segment (D-051).

---

## 6. DS / ML rigor highlights

- **Gold-set construction:** natural labels mined from LangChain Forum accepted answers that link to a
  docs page (hand-verified), deliberately kept as a separate slice from LLM-generated synthetic
  questions. Rebalanced to synthetic-primary + real slice after real links proved sparse (D-025).
- **Per-slice reporting, never blended** — the two slices disagree about what "better retrieval" means,
  so blending them hides the disagreement (D-028).
- **Ablation-driven, not vibes-driven:** 4 pipelines (dense / hybrid / +rerank / +decomposition) all
  still runnable via `--pipeline` as documented ablations rather than deleted code.
- **Negative results reported, not buried:**
  - **Cross-encoder reranking REJECTED** — 14× latency for a net loss; a 4-model bake-off confirmed
    *no* reranker beat the hybrid RRF baseline on real data (D-030/D-031/D-032).
  - **Query decomposition REJECTED** — real-slice recall@5 collapsed 0.404 → 0.173 at 5× latency (D-037).
  - **Pre-synthesis relevance grading REJECTED** — the grader inherited the LLM's own prior ("I already
    know the capital of France") and couldn't distinguish "I know this" from "the passages say this."
    Replaced with post-synthesis grounding verification (D-038).
- **Reproducibility:** fixed seeds, a fixed judge model, and every results file records its exact
  config, git SHA, and judge/synthesis model — any number in the README traces back to a specific run.
- **Honest drift reporting:** the 70B-vs-8B faithfulness/relevancy drift is reported both ways rather
  than picking the better-looking number (D-052).

Reproduce:
```
./tasks.ps1 eval --pipeline hybrid      # retrieval metrics -> results/eval_hybrid_<ts>.json
./tasks.ps1 ragas --sample 12 --seed 13 # RAGAS answer-quality -> results/ragas_<ts>.json
```

---

## 7. Hallucination handling (the safety story)

Grounding is enforced **structurally, not just measured**. A post-synthesis verifier (Groq 8B) checks
whether the drafted answer's claims are supported by its cited passages. If not: **one bounded
self-correction retry** (re-synthesize over the *same* context with corrective feedback — not a fresh
retrieval, since re-retrieval was already measured to hurt real questions), then **refusal**
("I don't know based on the provided documentation") rather than shipping an unsupported answer.

**Measured case:** on a weak-retrieval real question, Gemini 2.5 Flash correctly refused while Groq 70B
initially confabulated a plausible-looking API call — the 8B verifier caught it, triggering a retry
that produced a grounded answer instead (D-046/D-050).

Citations are numbered `[n]`, resolved and validated post-hoc; **invalid citations are surfaced in the
response (`invalid_citations`), never silently dropped**.

---

## 8. Decisions worth defending in an interview

| ID | Decision | The one-line why |
|---|---|---|
| D-016 | MIT `langchain-ai/docs` as corpus source | licensing verified; git clone of `.mdx` source beats scraping |
| D-018 | Labels from LangChain Forum, not GitHub Discussions | Discussions rarely link a docs page; Forum accepted answers do |
| D-021 | Qdrant cosine + normalized vectors, uuid5 ids, bge query instruction | idempotent re-index; bge requires the query prefix to perform |
| D-026/27 | Client-side RRF, k=60 | `qdrant-client` API lacked server-side fusion; k=60 is the published default, kept tunable |
| D-028/31 | **Ship hybrid despite it losing the real slice** | wins aggregate + the larger slice; the real-slice weakness is a *ranking* problem the grounding/refuse loop already defends against — a wrong-context answer gets caught by verification, not silently shipped. Explicit logged tradeoff, not an oversight. |
| D-030/32 | Reranker rejected after a 4-model bake-off | 14× latency, net-negative on real questions |
| D-037 | Query decomposition rejected | real recall@5 0.404 → 0.173 |
| D-038 | Grounding = post-synthesis verification, not pre-grading | pre-graders inherit the model's prior; verify the *draft's claims* against passages instead |
| D-041 | Self-correction = bounded re-synthesis with feedback, not re-retrieval | re-retrieval repeats a known failure mode; default 1 retry |
| D-042 | Real `ragas` library via our own gateway (not `langchain-groq`) | keeps caching/backoff/routing uniform; free-tier walls handled explicitly |
| D-043 | SSE streams the *verified* answer only | verification can retract the draft — never stream what you might retract |
| D-046 | Synthesis default flipped Gemini → Groq 70B, selectable per request | Gemini's 20 req/day free cap can't back a public demo |
| D-047 | Bake models into the Docker image | HF cold-start download exceeded Cloud Run's request timeout |
| D-048 | reindex task + Cloud Scheduler keep-alive | Qdrant free clusters idle-delete at ~4 weeks |
| D-052 | Final RAGAS re-run uses the *deployed* 70B, not the old 8B override | eval must measure what actually ships; drift documented |

---

## 9. Tech stack

Python 3.11+ (backend/agent/eval) · React + TypeScript / Vite (frontend) · **LangGraph** (agent
orchestration) · **Qdrant Cloud** (dense + BM25 sparse, client-side RRF) · **BAAI bge-small-en-v1.5**
(embeddings, 384-dim) · **Groq** (Llama 3.1 8B / 3.3 70B) + **Gemini 2.5 Flash** (routed through one
gateway) · `ragas==0.4.3` · **Langfuse** (observability) · **FastAPI** + SSE · **Docker** on **Cloud Run**
· **Vercel** · pydantic-settings · ruff · PowerShell task runner.

---

## 10. Honest caveats (carry these when quoting numbers)

1. **Real-question retrieval (n=26) is small** — directional, not conclusive. But it's the only slice
   with genuinely independent natural labels (not generated by the model being evaluated).
2. **Hybrid ships as default despite losing the real slice** — an explicit, logged tradeoff (D-028/D-031).
3. **RAGAS numbers are judge-context-capped** to fit Groq's free-tier TPM limit, biasing faithfulness
   down relative to the agent's actual 8-chunk context.
4. **The 70B-vs-8B drift (D-052) is measured on n=12** and could be noise — reported both ways.
5. **Free tier only, by design** — bounds throughput (Gemini 20 req/day) and cluster durability
   (Qdrant idle-delete); both mitigated, not eliminated.

---

## 11. Keyword bank (for ATS / JD matching)

RAG · Retrieval-Augmented Generation · agentic AI · AI agents · LangGraph · LangChain · hybrid retrieval ·
dense retrieval · sparse retrieval · BM25 · Reciprocal Rank Fusion (RRF) · vector database · Qdrant ·
embeddings · sentence-transformers · cross-encoder reranking · query decomposition · semantic search ·
chunking · LLM · Groq · Llama 3 · Gemini · prompt engineering · grounding verification · hallucination
mitigation · citations · self-correction · evaluation harness · offline evaluation · ablation study ·
gold dataset · recall@k · MRR · nDCG · RAGAS · faithfulness · answer relevancy · context precision/recall ·
LLM-as-judge · observability · Langfuse · tracing · distributed tracing · FastAPI · REST API · SSE ·
server-sent events · streaming · React · TypeScript · Vite · Docker · Google Cloud Run · Cloud Scheduler ·
Vercel · CI/CD · serverless · scale-to-zero · rate limiting · exponential backoff · caching · CORS ·
Python · pydantic · pytest · reproducibility · MLOps · disaster recovery.

---

## 12. Scale/scope facts (for "impact" phrasing)

- 11,035 chunks / 751 docs pages indexed across 2 Qdrant collections.
- 126-question gold set; 4 retrieval pipelines benchmarked; 5 RAGAS metrics.
- 52 logged decisions; ~80+ tests; 10 build layers, each gated.
- 3 LLM models routed through 1 gateway; 6 free-tier services integrated
  (Groq, Gemini, Qdrant, Cloud Run, Vercel, Langfuse) — **$0 running cost**.
- Public URL, live, scale-to-zero.

---

## 13. Pre-styled resume bullets (LaTeX, `resumeProjectHeading`)

Two ready-to-drop variants. Same project, different lead: SDE variant leads with deployment/
architecture/latency; DS/ML variant leads with eval rigor and the rejected-ablation negative result.

### Variant A — SDE-focused

```latex
\resumeProjectHeading
    {\href{https://github.com/mahendra-kausik/docsGPT}{\textbf{\large{\underline{DocsGPT-Agent: Cited Agentic RAG over Developer Docs}}} \href{https://github.com/mahendra-kausik/docsGPT}{\raisebox{-0.1\height}\faExternalLink }} $|$ \large{\underline{Python, LangGraph, Qdrant, FastAPI, React/TS}}}{}
    \resumeItemListStart
      \resumeItem{\normalsize{Deployed a cited agentic QA system over 11,035 chunks of LangChain docs to a public URL: a LangGraph state machine (retrieve $\rightarrow$ synthesize $\rightarrow$ verify $\rightarrow$ retry/refuse) on Cloud Run (scale-to-zero) with a React/TypeScript UI on Vercel, running entirely on free-tier infrastructure.}}

      \resumeItem{\normalsize{Built hybrid retrieval (bge-small dense + BM25 with Reciprocal Rank Fusion) on Qdrant reaching \textbf{0.693 recall@5} (+13\% over dense) at \textbf{579 ms p50} end-to-end latency; streamed lifecycle events and verified answer tokens over SSE.}}

      \resumeItem{\normalsize{Routed every LLM call (Groq / Gemini) through a single gateway with exponential backoff+jitter, response caching, and model routing; hardened durability with a one-command reindex and traced every agent node + LLM call in Langfuse.}}

      \resumeItem{\normalsize{Enforced grounding structurally: a post-synthesis verifier catches unsupported claims and drives a bounded self-correction retry that refuses rather than hallucinate, with per-claim citations validated post-hoc.}}
    \resumeItemListEnd
```

### Variant B — DS/ML-focused

```latex
\resumeProjectHeading
    {\href{https://github.com/mahendra-kausik/docsGPT}{\textbf{\large{\underline{DocsGPT-Agent: Cited Agentic RAG over Developer Docs}}} \href{https://github.com/mahendra-kausik/docsGPT}{\raisebox{-0.1\height}\faExternalLink }} $|$ \large{\underline{Python, LangGraph, Qdrant, RAGAS, FastAPI}}}{}
    \resumeItemListStart
      \resumeItem{\normalsize{Built a quantitative retrieval eval harness over a 126-question gold set (26 human-verified LangChain-Forum answers + 100 synthetic); benchmarked 4 pipelines per-slice (dense / hybrid / rerank / decomposed) with recall@k, MRR, and nDCG.}}

      \resumeItem{\normalsize{Shipped hybrid retrieval (dense bge-small + BM25 RRF) reaching \textbf{0.693 recall@5} (+13\% over dense); \textbf{measured and rejected} cross-encoder reranking and query decomposition after both showed net-negative recall on real questions at 5--14$\times$ latency, and traced the gap to lexical overlap in synthetic labels.}}

      \resumeItem{\normalsize{Quantified answer quality with a RAGAS harness (fixed Groq-8B judge, fixed seeds, per-metric $n$ reported) measuring faithfulness, relevancy, and correctness; reported honest absolute numbers and known judge-context bias rather than cherry-picking.}}

      \resumeItem{\normalsize{Reduced hallucination via a LangGraph agent that verifies grounding post-synthesis and refuses unsupported answers; deployed the full pipeline to Cloud Run + Vercel with per-node Langfuse tracing, on free-tier infrastructure.}}
    \resumeItemListEnd
```

**Note:** +13% is the aggregate (dense 0.612 → hybrid 0.693). If probed, the honest per-slice story
is hybrid +24% on synthetic but dense winning the real slice (see §3) — documented, defensible.
