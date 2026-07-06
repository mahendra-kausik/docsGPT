# DECISIONS.md — Decision Log

> Every non-trivial choice, with the reasoning to defend it in an interview.
> **Claude Code: append a new entry whenever you make a decision a reviewer could question.**
> Newest entries go at the bottom. Never silently rewrite a past decision — supersede it with a new entry.

### Entry template (copy this)

```
## D-XXX — <short title>
- **Date / Layer:** <when>
- **Context:** <what problem/choice prompted this>
- **Decision:** <what we chose>
- **Why:** <the core reason(s)>
- **Alternatives considered:** <options + why not>
- **Tradeoffs / risks:** <what we give up; what could go wrong; how we'd mitigate>
- **Supersedes:** <D-YYY if applicable>
```

---

## D-001 — Project shape: agentic, cited, evaluated docs-QA (not "chat with PDF")
- **Date / Layer:** Pre-build.
- **Context:** Need a placement project that stands out from the generic single-pass PDF-chat every student builds, and that defends well for **both** SDE and DS interviews.
- **Decision:** Build a deployed agentic RAG system over developer docs with hybrid retrieval, reranking, per-claim citations, a quantitative eval harness, and observability.
- **Why:** The differentiators (multi-hop agent, hybrid+rerank, citations, eval) are exactly what naive RAG lacks, and on a docs corpus they produce *measurable* gains. Engineering + DS rigor in one artifact.
- **Alternatives considered:** (a) Single-pass "chat with PDF" — too common, no measurable edge. (b) Pure benchmark project on HotpotQA/2WikiMultiHop — reads as "numbers on a dataset," not a product. (c) Biomedical/PubMed QA (BioASQ) — strongest gold labels but risks looking like a benchmark; kept as a fallback.
- **Tradeoffs / risks:** More moving parts than a simple RAG demo; mitigated by strict layer-by-layer build and building the eval harness early so every part is measured.

## D-002 — Corpus: LangChain/LangGraph docs + GitHub (Kubernetes as alternative)
- **Date / Layer:** Pre-build (finalize in Layer 1).
- **Context:** Need a corpus where retrieval genuinely matters, multi-hop is natural, quality is measurable, and licensing is safe.
- **Decision:** Primary = LangChain/LangGraph docs + GitHub Discussions/Issues + changelogs. Alternative = Kubernetes docs (CC BY 4.0).
- **Why:** Deep personal familiarity (better eval questions, fluent interview answers); fast-moving → real version-drift → genuine multi-hop ("what changed between versions"); GitHub Discussions provide natural relevance labels.
- **Alternatives considered:** Kubernetes (cleaner license, but more stable → weaker version-drift story, larger to wrangle); Stripe/Next.js (licensing/ToS less clear for scraping).
- **Tradeoffs / risks:** Docs-site licensing must be checked before scraping (⚠️). GitHub API is rate-limited (5,000 req/hr authenticated) → must cache. If licensing is unclear, switch to Kubernetes (CC BY 4.0).

## D-003 — Gold labels come mainly from GitHub *Discussions*, not *Issues*
- **Date / Layer:** Layer 3.
- **Context:** The "accepted answers = free relevance labels" idea needs scrutiny.
- **Decision:** Use GitHub **Discussions'** native "marked as answer" as the primary natural label source; for Issues, hand-map the resolving comment/PR. Target ~100–150 hand-verified Q→gold-chunk pairs + RAGAS-synthetic augmentation.
- **Why:** **GitHub Issues have no "accepted answer" feature — Discussions do.** Assuming Issues give free labels would be wrong and would undermine the eval's credibility.
- **Alternatives considered:** Fully synthetic gold set (RAGAS only) — cheaper but weaker validity; kept as augmentation, not the core.
- **Tradeoffs / risks:** Manual curation costs real time in Week 1. Mitigation: start the gold set small (~100) and grow it; prefer natural labels for the headline numbers.

## D-004 — Vector store: Qdrant Cloud free tier (dense + sparse + RRF in one)
- **Date / Layer:** Layer 2.
- **Context:** Need a *deployed* (not localhost) store that does dense + keyword retrieval with fusion, on free tier.
- **Decision:** Qdrant Cloud free tier; dense vectors + BM25 sparse vectors + server-side RRF fusion in a single collection.
- **Why:** One engine for hybrid retrieval → no separate Elasticsearch to run/deploy. Free, no credit card, ~1M vectors @ 768-dim — ample for our corpus. Highest leverage for a time-boxed build.
- **Alternatives considered:** pgvector on Cloud SQL ("one database" story, but not free beyond trial → spends credits); Chroma (weak for deployed/persistent multi-tenant); Weaviate/Vertex Vector Search (heavier / cost risk).
- **Tradeoffs / risks:** ⚠️ **Free clusters auto-suspend after ~1 week inactivity and delete after ~4 weeks.** Mitigation: keep the chunked corpus as JSONL in the repo/bucket (durable source of truth), a one-command re-index script (Layer 8), and a scheduled keep-alive ping. Never rely on the free cluster as the only copy of the data.

## D-005 — Embeddings: BAAI BGE, self-hosted (bge-small default, bge-base if needed)
- **Date / Layer:** Layer 2.
- **Context:** Thousands of embed calls during indexing + eval; must be free and deployable on CPU.
- **Decision:** `bge-small-en-v1.5` (384-dim) as default; benchmark against `bge-base-en-v1.5` (768-dim) on our gold set and pick the winner.
- **Why:** Free forever, CPU-friendly, no per-call cost; the small/base ablation is itself a defensible DS talking point. MTEB rankings don't predict our corpus → we measure on our own data.
- **Alternatives considered:** Gemini embeddings (zero local compute, but free tier may train on inputs — acceptable for *public* docs only; log if used); OpenAI embeddings (paid).
- **Tradeoffs / risks:** Self-hosting adds a little container weight/latency; acceptable on Cloud Run CPU. Keep embedding dimension consistent between index and query (a classic bug).

## D-006 — Keyword retrieval + fusion: Qdrant native BM25 sparse + RRF (k=60)
- **Date / Layer:** Layer 4.
- **Context:** Docs queries are entity/token-heavy (API names, flags, versions) where dense alone underperforms.
- **Decision:** BM25 sparse vectors in Qdrant fused with dense via RRF, `k=60` (`score(d)=Σ 1/(k+rank_i(d))`).
- **Why:** BM25 anchors exact-token matches; dense captures paraphrase; RRF combines them robustly without score-scale tuning. Server-side in Qdrant = minimal code.
- **Alternatives considered:** `rank_bm25` in-memory (fine <50k chunks, but a second system to manage); Postgres full-text; weighted score fusion (needs scale normalization — more brittle than RRF).
- **Tradeoffs / risks:** RRF `k` is a hyperparameter; default 60 is standard — if tuned, log the sweep and result.

## D-007 — Reranker: BGE cross-encoder, self-hosted on CPU
- **Date / Layer:** Layer 4.
- **Context:** Fused candidate ordering still misranks on long-tail phrasing; a cross-encoder fixes precision@k.
- **Decision:** `bge-reranker-base` (or `-v2-m3`); fall back to `ms-marco-MiniLM-L-6-v2`/FlashRank if CPU latency is tight. Rerank top ~50 fused candidates → keep top ~5–8.
- **Why:** Cross-encoders measurably lift precision on entity-heavy queries; open, free, runs on CPU in-container. Measure and report rerank latency.
- **Alternatives considered:** Cohere Rerank (good, but free/trial quota ⚠️ ~1k calls/month too tight for a live demo — optionally used only for a one-off API-vs-open ablation).
- **Tradeoffs / risks:** Adds latency per query; bounded by capping candidate count. Report the latency honestly (e.g., "~90 ms for 50 candidates on CPU").

## D-008 — LLMs: Groq 8B for cheap/high-volume nodes + Gemini Flash for synthesis, routed
- **Date / Layer:** Layer 5–6.
- **Context:** Agentic queries make 5–15 LLM calls; free-tier daily quotas are the binding constraint.
- **Decision:** Route **query decomposition / relevance grading / query rewriting** and **RAGAS judge calls** to **Groq `llama-3.1-8b-instant`** (⚠️ ~14,400 RPD — huge headroom, fast). Route **final answer synthesis** to **Gemini Flash** (Groq 70B as fallback, ⚠️ ~1,000 RPD). All calls through one wrapper with exponential backoff + jitter and a response cache.
- **Why:** Concentrating high-volume calls on Groq 8B's large daily budget is what keeps a live demo inside free limits while reserving the stronger model for the answer the user sees. This routing is itself a resume-worthy production-maturity point.
- **Alternatives considered:** All-Gemini (⚠️ Pro is now paid-only; Flash RPD is small and volatile — would throttle the agent); all-Groq (open models only — fine, kept as fallback); paid tier (violates free-tier constraint).
- **Tradeoffs / risks:** ⚠️ Free-tier limits shift — **verify live in AI Studio / Groq console and don't hard-code numbers.** Gemini free tier may train on inputs (fine for public-docs answers; don't send private data). Backoff + cache are mandatory, not optional.

## D-009 — Agent framework: LangGraph
- **Date / Layer:** Layer 5.
- **Context:** The differentiator is a *genuine* cyclic agent (decompose → retrieve → grade → rewrite → generate → verify → self-correct), not single-pass RAG.
- **Decision:** LangGraph for orchestration.
- **Why:** First-class support for cycles, state, streaming, and human-in-the-loop; renders cleanly in Langfuse traces; short ramp given existing LangChain familiarity. Makes the "agentic" claim demonstrable via the trace.
- **Alternatives considered:** LlamaIndex Workflows/agents (great RAG modules — may borrow its retrievers under LangGraph); custom loop (more control, more plumbing, weaker tracing story); AutoGen (multi-agent overkill here).
- **Tradeoffs / risks:** Framework churn; pin the version. Keep nodes small and testable so the graph stays inspectable.

## D-010 — Evaluation: custom retrieval metrics + RAGAS, judge fixed, built early
- **Date / Layer:** Layer 3 (before hybrid/rerank/agent).
- **Context:** Numbers must be comparable across every change and defensible in interviews.
- **Decision:** Custom Recall@k / MRR / nDCG / Hit Rate vs gold + RAGAS (faithfulness, answer relevancy, context precision/recall, answer correctness). **Fix the RAGAS judge model (Groq 8B) across all runs**; record config + judge in every results file. Build the harness in Layer 3 and re-run it after Layers 4 and 5.
- **Why:** Retrieval metrics need no LLM judge → maximally defensible; RAGAS adds answer-quality/hallucination signal. Fixing the judge makes ablations valid. Building early turns every later commit into a measured delta.
- **Alternatives considered:** Eval only at the end (loses the delta story); RAGAS-only (no clean retrieval numbers); rotating judge models (makes runs incomparable).
- **Tradeoffs / risks:** RAGAS scores are judge-dependent and can show artifacts (near-1.0 faithfulness with low relevancy when context is thin) → always read faithfulness with relevancy; disclose the judge. Synthetic gold has validity limits → hand-verify a core subset.

## D-011 — Observability: Langfuse Cloud free tier
- **Date / Layer:** Layer 7.
- **Context:** Need per-node tracing, cost, and latency for a call-heavy agent — and a story for interviews.
- **Decision:** Langfuse Cloud free tier (⚠️ ~50k observations/month, ~30-day retention); every LangGraph node emits a span.
- **Why:** Native LangGraph integration, nested spans, token cost + latency, scores/eval data model; OpenTelemetry-based (no lock-in).
- **Alternatives considered:** Arize Phoenix (open-source, strong offline RAG-eval viz — good complement); raw logging (no cost/latency aggregation, weak demo).
- **Tradeoffs / risks:** Free observation cap ⚠️ — sample or cap tracing in high-volume eval runs so you don't exhaust it.

## D-012 — Deployment: Cloud Run (FastAPI) + Vercel (React) + Qdrant Cloud; GCP credits as insurance
- **Date / Layer:** Layer 8.
- **Context:** Must be a public URL on free tier; agent + reranker run server-side.
- **Decision:** FastAPI container on Cloud Run (always-free tier, scales to zero); React on Vercel Hobby; Qdrant Cloud; secrets in Secret Manager. GCP credits reserved for a warm instance, a pgvector alternative, or a one-off embedding batch.
- **Why:** All-free public deployment; Cloud Run scale-to-zero keeps cost at $0; managed Qdrant satisfies "deployed vector DB."
- **Alternatives considered:** Hugging Face Spaces (viable, but Cloud Run showcases real cloud/container skills and uses existing GCP); single VM (always-on cost; more ops).
- **Tradeoffs / risks:** ⚠️ Scale-to-zero → **cold starts** (warm or min-instance for demos, spends a little credit). ⚠️ Vercel Hobby is **non-commercial** (fine for portfolio). ⚠️ Cloud Run always-free is region-specific — deploy in an eligible region. Add the Qdrant re-index + keep-alive here (see D-004).

---

<!-- New decisions appended below by Claude Code as the build proceeds -->

## D-013 — Task runner: PowerShell `tasks.ps1` (supersedes the Makefile)
- **Date / Layer:** Layer 0.
- **Context:** PROJECT_PLAN's Layer 0/8 gates call for `make setup`, but the dev machine is Windows 11 with no native `make` (only Windows PowerShell 5.1; no `pwsh`).
- **Decision:** Ship a PowerShell `tasks.ps1` exposing `setup | test | lint | format | help` instead of a Makefile. Run as `./tasks.ps1 setup`.
- **Why:** Same short-verb ergonomics with zero extra tooling to install on Windows; the app still deploys to a Linux container at Layer 8, so this only affects local dev.
- **Alternatives considered:** Makefile + installing `make` (adds a Windows dependency); a Python `tasks.py`/`invoke` runner (cross-platform but adds a dep and indirection); keeping both (more surface to maintain).
- **Tradeoffs / risks:** Reads `make setup` in the plan as `./tasks.ps1 setup`. CI runs the equivalent steps directly (no task runner) on Linux. If we ever need a non-Windows contributor, add a thin `make`/`tasks.py` shim.
- **Supersedes:** the Makefile reference in PROJECT_PLAN §4 (Layer 0) and §Layer 8 gate wording.

## D-014 — Dependencies: `pyproject.toml` (editable package) + pinned `requirements.txt`
- **Date / Layer:** Layer 0.
- **Context:** Modules live under `src/{ingest,retrieval,agent,eval,api}` and must import cleanly (`from src.retrieval import ...`); CLAUDE.md §4 also mandates pinned versions for reproducibility.
- **Decision:** `pyproject.toml` declares the project as an installable package (setuptools; `pip install -e .`) with abstract dep ranges; `requirements.txt` holds the EXACT pins (the reproducible lock). `setup` installs the editable package then the pins.
- **Why:** Editable install removes all `sys.path` hacks and makes tests/CI import the package identically; exact pins guarantee the eval harness reproduces its numbers.
- **Alternatives considered:** `requirements.txt` only (needs path hacks, no clean package); `uv` (fast, lockfile — but another tool to install and defend on a portfolio project); Poetry (heavier).
- **Tradeoffs / risks:** Deps declared in two places can drift; mitigated by treating `requirements.txt` pins as the authority (installed last) and keeping pyproject ranges loose.

## D-015 — Lean, layer-incremental dependencies; config via pydantic-settings
- **Date / Layer:** Layer 0.
- **Context:** Installing the full ML stack (torch, sentence-transformers, qdrant-client, langgraph, ragas, fastapi) just to scaffold would make `setup` slow/fragile on Windows and bloat the env before it's needed.
- **Decision:** Layer 0 pins only scaffold deps (pydantic, pydantic-settings, pyyaml, python-dotenv, pytest, ruff). Heavy deps are added in the layer that first needs them. Config loads through a single typed `Settings` (pydantic-settings): secrets from `.env`, tunables from `config.yaml` (config.yaml wins for tunables via init-kwarg precedence).
- **Why:** Fast, defensible setup; one typed config surface makes ablations a one-line change (CLAUDE.md §6) and keeps secrets out of `config.yaml`/git.
- **Alternatives considered:** Install everything up front (slow, brittle, premature); plain `os.getenv` + constants (scattered magic numbers, no typing — violates §6); a bare dataclass config (loses env/.env layering that pydantic-settings gives free).
- **Tradeoffs / risks:** `requirements.txt` grows layer by layer — each addition must stay pinned. Tunables duplicated between `.env.example` and `config.yaml`; documented that `config.yaml` is the authority for tunables.
- **Environment note:** local Python is **3.13.3** (plan floor was 3.11+); CI pinned to 3.13 to match. Watch for 3.13 wheel availability on heavy ML deps at Layer 2 (torch/sentence-transformers ship 3.13 wheels; verify at that layer).

## D-016 — Corpus source: ingest the MIT `langchain-ai/docs` repo (.mdx), Python-focused
- **Date / Layer:** Layer 1a.
- **Context:** D-002 chose LangChain/LangGraph docs but left the exact source + scope open, pending a licensing check (PLAN §1 flags this as mandatory before scraping).
- **Decision:** Ingest the **`langchain-ai/docs`** repo directly (shallow `git clone`, SHA pinned in `data/corpus/manifest.json`) rather than scraping the rendered site. Scope = **Python-focused OSS**: `src/oss/{python,langchain,langgraph,deepagents,concepts,integrations}` + top-level `src/oss/*.mdx`. Exclude JavaScript, LangSmith, `src/snippets` (used only for inlining), `reference`, `contributing`.
- **Why:** Verified licensing — the docs repo is **MIT** (© 2025 LangChain; MIT explicitly covers "associated documentation files"), langchain/langgraph repos are MIT, and `docs.langchain.com/robots.txt` sets `Content-Signal: ai-train=yes, search=yes, ai-input=yes`. The repo's `.mdx` source is clean, versioned, and rate-limit-free vs HTML scraping. Python-only avoids near-duplicate JS/Py chunks that would muddy Recall@k and reranking (user works in Python).
- **Alternatives considered:** Scrape rendered `docs.langchain.com` (ToS/robots ambiguity, dirtier HTML); include JavaScript (near-duplicate answers pollute retrieval); include LangSmith (its site ToS forbids reverse-engineering/derivative works — weaker licensing story than MIT OSS); Kubernetes CC-BY fallback (unneeded now that LangChain licensing is clean).
- **Tradeoffs / risks:** The repo is a *current* snapshot, so the "version-drift multi-hop" story (D-002) is limited to the migration/`releases`/changelog pages it contains (they exist — good enough). Snapshot SHA `662d399`. Re-run rebuilds from the pinned SHA; the raw clone is git-ignored, the chunked JSONL is the committed source of truth (D-004).

## D-017 — Hand-rolled, fence-aware MDX cleaning + structure-aware chunking
- **Date / Layer:** Layer 1a.
- **Context:** MDX mixes YAML frontmatter, JS-style imports/exports, JSX components, Mintlify `:::lang` conditional-content directives, base64 `<img>` data URIs, and code fences. A naive Markdown parser mangles code and can't do "keep Python, drop JS."
- **Decision:** A hand-rolled cleaner (`src/ingest/mdx.py`) walks lines with explicit **code-fence tracking**, so MDX-isms are stripped ONLY outside code: frontmatter → `title`; drop `import … from '…'`/`export …`; strip JSX tags but keep inner text; `:::python` kept / `:::js` dropped / admonitions kept (markers stripped, 3+ colons + inline content handled); `<img>`/base64 data URIs removed. Chunking (`chunker.py`) splits into heading/code/text blocks, packs to `chunk_size` chars, **never splits a code fence** (oversize code → its own chunk), carries prose-only overlap, tags each chunk with a heading breadcrumb, and drops near-empty (heading-only/lone-image) chunks below `min_chunk_chars`.
- **Why:** Fence-aware line walking is the simplest way to *guarantee* the gate's "code intact" requirement and enables language-selective content; hand-rolling keeps deps lean (D-015) and fully controllable/defensible. Result: **11,035 chunks / 751 files @ sha 662d399**, ~11 MB, unique ids, all fields present, no base64/`:::` blow-ups.
- **Alternatives considered:** `markdown-it-py`/`mistune`/full MDX AST (heavier deps, JSX confuses them, harder to guarantee code-intactness and language filtering); LangChain's own `MarkdownHeaderTextSplitter` (not MDX/JSX/`:::`-aware); keeping JS variants (duplicate noise).
- **Tradeoffs / risks:** Heuristic cleaning has a long tail: ~5 chunks keep large base64 strings that live *inside* doc code examples (ollama/scrapeless) — left intact per the "never touch code" rule; flagged for Layer 2 (bge-small will truncate them at embed time). 2 benign `:::` residues remain inside Mermaid code. Community `integrations/providers/*` pages occasionally show code-fences-inside-code (odd-fence artifact). Snippet transclusion is best-effort, recursion-guarded, ≤3 levels.

## D-018 — Natural gold labels: LangChain Forum (Discourse), not GitHub Discussions
- **Date / Layer:** Layer 1b.
- **Context:** D-002/D-003 assumed GitHub Discussions' "marked as answer" would supply natural relevance labels. On checking live: `langchain-ai/langchain` has only **4 discussions (all Announcements, none answered)** and one pinned "**Discussions have moved to the LangChain Forum!**" — GitHub Discussions were migrated to `forum.langchain.com` in mid-2025; `langchain-ai/langgraph` never had Discussions.
- **Decision:** Source natural labels from the **LangChain Forum** (Discourse) instead. Use **solved topics** (accepted answer) from the **OSS Product Help** category — **163 usable question→accepted-answer pairs** (well within the 100–150 gold target). Fetch via the public Discourse JSON API (`/categories.json`, `/c/{slug}/{id}.json`, `/t/{id}.json`), no auth, respecting robots.txt (never `/search`), disk-caching every topic.
- **Why:** The plan's "natural labels" premise survives the migration — it just moved platforms. The forum exposes a clean `accepted_answer` marker per post. Verified feasibility (OSS Product Help ~165 solved) before committing.
- **Alternatives considered:** GitHub closed Issues hand-mapped (~10k, no accepted-answer markers → far more manual, noisier); synthetic-only gold (loses real-question distribution); Kubernetes corpus (throws away Layer 1a). The GitHub-Discussions GraphQL fetcher built earlier was removed once found empty.
- **Tradeoffs / risks:** Forum is newer/smaller than the imagined GitHub trove; ~120 of the labels are Python (`python-help`), a few JS. **Updates the label-source half of D-002/D-003** (corpus choice unchanged). ⚠️ **CLAUDE.md §7 flag raised to the user before proceeding** — plan assumption was outdated.

## D-019 — Forum data used as GOLD-EVAL SEEDS ONLY (not corpus content)
- **Date / Layer:** Layer 1b.
- **Context:** Forum posts are user-generated with **no explicit reuse license** (the forum's `/tos` is the commercial LangSmith SaaS agreement, not a content licence). Question: put forum Q&A into the retrieval corpus, or use only for eval?
- **Decision:** Use solved topics as **gold-eval seeds only**. A topic's **question** becomes an eval item (attributed by URL); its gold **target** is the MIT docs chunk(s) that answer it (mapped at Layer 3). The retrieval **corpus stays MIT-docs-only**. Committed `data/gold/forum_seeds.jsonl` carries question + metadata + accepted-answer URL but **no answer body**; full answer text lives only in the git-ignored raw cache (`data/raw/forum/`) for local mapping.
- **Why:** Two independent reasons, decided with the user. (1) **Methodology:** putting a gold answer verbatim into the corpus creates **test-set leakage** (retrieval becomes trivial exact-match → inflated, indefensible Recall@k). Grading real questions against docs chunks keeps one clean retrieval space and upgrades the eval from synthetic to real-query distribution. (2) **Licensing:** ships nothing unlicensed — corpus = MIT, eval questions = attributed real user questions.
- **Alternatives considered:** Also ingest forum Q&A as `type='forum'` corpus chunks (more real-phrasing coverage, but leakage risk + redistributes unlicensed text); skip forum entirely for synthetic-only (loses real labels). If forum content is ever wanted in the corpus as a deliberate product choice, it must be a disclosed, leakage-guarded decision (hold out tested questions).
- **Tradeoffs / risks:** Questions only answerable from forum (not docs) become out-of-scope and are dropped/flagged as corpus gaps at Layer 3 — correct for a docs-QA system. A low dense-only baseline (Layer 3) is expected input to Layers 4–5, **not** a reason to inject answers into the corpus.

## D-020 — Free-tier limits re-verified (Jul 2026); Gemini synthesis model-id updated (2.0 Flash retired)
- **Date / Layer:** Layer 2 prep (2026-07-06).
- **Context:** CLAUDE.md §4/§7 require verifying free-tier limits *live* before relying on them; PROGRESS.md flagged this as an open blocker. Re-checked Qdrant, Cloud Run, Groq, and Gemini against each provider's own docs.
- **Decision:** Record verified numbers and adjust two things:
  - **Qdrant Cloud free:** 1 GB RAM / 0.5 vCPU / 4 GB disk / 1 node (~1M vec @768-dim); **suspend after 1 wk idle, delete after 4 wk** — matches D-004. Free cluster **created on GCP / us-central1** (co-located with the planned Cloud Run region); endpoint auth-verified (`/collections` → 200 OK).
  - **Cloud Run always-free:** 180k vCPU-s + 360k GiB-s + 2M requests / month, region-specific — matches D-012. Keep `GCP_REGION=us-central1`.
  - **Groq free `llama-3.1-8b-instant`:** 30 RPM / **14,400 RPD**, **plus a tokens-per-day (TPD) dimension now enforced**. Request headroom is huge; **TPD is the real binding cap** for heavy eval runs. Confirm exact org caps at `console.groq.com/settings/limits`.
  - **Gemini:** `gemini-2.0-flash` was **retired 2026-06-01**. Switch synthesis to a current Flash (default `gemini-2.5-flash`; newer Gemini 3.x Flash available) — free tier ≈ 10 RPM / 250k TPM / 1,500 RPD. `SYNTHESIS_MODEL` in `.env.example` updated.
- **Why:** Free-tier numbers shift; the plan mandates live verification, not hard-coded values. D-008's routing rationale (concentrate volume on Groq 8B, reserve the stronger model for the user-facing answer) still holds — only the Gemini model id changed, and Groq's binding constraint shifted from *requests* to *tokens/day*.
- **Alternatives considered:** Keep the older hard-coded numbers (violates §4 "don't hard-code, verify live"); all-Groq synthesis (retained as fallback per D-008).
- **Tradeoffs / risks:** Groq TPD could throttle large eval runs → mitigate with the mandated response cache + backoff and by capping RAGAS sample size. Gemini free tier may train on inputs (public docs only — acceptable). **Not yet re-verified:** Vercel Hobby and Langfuse caps — deferred to their layers (7/8).
- **Supersedes:** the model-id portion of **D-008** (`gemini-2.0-flash` → current Flash); the routing strategy in D-008 is unchanged.

## D-021 — Dense index spec: Qdrant cosine, normalized bge, model-derived dim, full payload, deterministic UUID ids
- **Date / Layer:** Layer 2 (2026-07-06).
- **Context:** Layer 2 needs the chunked corpus embedded and pushed to the deployed Qdrant cluster with a query-time search fn. Several small but questionable choices: distance metric, where the embedding dim comes from, what to store per point, and how to map our string chunk ids to Qdrant point ids.
- **Decision:**
  - **Cosine distance + L2-normalized embeddings** (`normalize_embeddings=True`). bge vectors are trained for cosine similarity.
  - **Embedding dimension is read from the model** (`get_embedding_dimension`), not hard-coded — so the bge-small(384)↔bge-base(768) ablation (D-005) needs no index-code change and index/query dims can never silently disagree.
  - **Store the full `Chunk` payload** (text + `source_url`/`heading_path`/provenance) on each point. 11k×~600 chars ≈ 7 MB vs the 4 GB free disk — trivial — and it removes the need for a second document store when reranking (Layer 4) and citing (Layer 5).
  - **Point ids = `uuid5(namespace, chunk.id)`** (deterministic). Qdrant requires uint64/UUID ids; our chunk ids are short hex hashes. uuid5 keeps upserts idempotent across re-indexes regardless of the chunk-id format.
  - **bge query instruction** (`Represent this sentence for searching relevant passages:`) is prepended to **queries only**, per documented bge-en-v1.5 usage; config-controlled (`query_instruction`, empty disables) so it travels with the model choice.
- **Why:** Each choice is the model-intended / least-surprising option and keeps future ablations to one-line config edits (CLAUDE.md §6). Storing the payload trades a little Qdrant disk (abundant) for a much simpler retrieval/citation path.
- **Alternatives considered:** Dot-product without normalization (equivalent once normalized, but normalize makes scores comparable and is bge's documented setup); id-only payload + separate doc store (extra moving part for no benefit at this scale); `int(chunk.id, 16)` point ids (works only if ids stay 16-hex — brittle); dropping the query instruction (measurably lower Recall on bge-en-v1.5).
- **Tradeoffs / risks:** Payload duplicates the corpus text in Qdrant — fine at this scale, and the JSONL remains the durable source of truth (D-004). If we later switch to a non-bge model, revisit the query instruction.

## D-022 — Embedding input hygiene: scrub base64/data-URI runs, then hard char-cap
- **Date / Layer:** Layer 2 (2026-07-06).
- **Context:** Layer-1a profiling (recorded in PROGRESS) found 2 chunks ~52k chars dominated by a single ~52k base64 run, plus ~3 more with 600–1,100-char base64 runs — the D-017 "base64 inside doc code examples" tail. Fed raw, these consume bge's entire 512-token budget on noise.
- **Decision:** Before encoding, `scrub_binary()` replaces `data:…;base64,…` URIs and any `[A-Za-z0-9+/]{200,}` run with `"[binary omitted]"`, then `prepare_text()` caps length at `embed_max_chars` (8000). Applied to both passages and queries.
- **Why:** Removes semantically empty bytes so the model embeds the real surrounding text; the 200-char threshold is well above normal code tokens/hashes (sha256 = 64) but far below the real base64 blobs, so ordinary code/prose is untouched (unit-tested).
- **Alternatives considered:** Drop the offending chunks entirely (loses the real code around the blob); rely only on tokenizer truncation (would keep the first 512 tokens of pure base64 → a garbage vector); scrub during ingestion/Layer-1a (violates D-017's "never mutate code" rule — better to scrub at embed time, leaving the JSONL faithful).
- **Tradeoffs / risks:** A pathological 200+ char run of legitimate base64-charset text (rare in prose/code — they contain spaces/punctuation) would be scrubbed; acceptable given the retrieval benefit.

## D-023 — torch CPU build (Windows local now; CPU wheel index for the Cloud Run image later)
- **Date / Layer:** Layer 2 (2026-07-06).
- **Context:** bge embedding + the Layer-4 reranker run on CPU (free tier, no GPU on Cloud Run). torch is the heavy dep.
- **Decision:** Pin `torch==2.12.1`; on Windows the PyPI wheel is already the CPU build (installed as `2.12.1+cpu`). At Layer 8 the Linux container must install torch from `https://download.pytorch.org/whl/cpu` to avoid the bundled-CUDA Linux wheel (~GBs) that we'd never use.
- **Why:** Keeps local dev and the eventual container CPU-only and lean, matching the free-tier deployment target (D-012).
- **Alternatives considered:** GPU torch (no free GPU; pointless); a heavier all-in-one image (slow cold starts on Cloud Run, D-012 risk).
- **Tradeoffs / risks:** Must remember the CPU index-url in the Layer 8 Dockerfile or the image balloons; flagged here so it isn't forgotten.

## D-024 — Gold-set build: batch propose → human review.md → compile; drop docs-unanswerable questions
- **Date / Layer:** Layer 3 (2026-07-06).
- **Context:** Layer 3 needs the 163 forum questions (D-018/D-019) mapped to the MIT-docs chunk(s) that answer them, hand-verified (PLAN §1/§3 gold-label reality check). Two sub-choices: the verification *workflow*, and what to do with questions the docs corpus can't answer.
- **Decision:**
  - **Batch, file-based verification** (`src/eval/propose.py` → `compile_gold.py`): for each question, retrieve a **20-candidate** dense pool, pair it with the accepted forum answer (read from the git-ignored raw cache), and emit `review.md` (human edits one `DECISION` line per question) + `candidates.jsonl` (machine rank→chunk_id map). The compiler resolves decisions into `data/gold/gold.jsonl`. Chosen over interactive question-by-question review (too slow for 163) and over auto-map-then-spot-check (weaker credibility) — decided with the user.
  - **Candidate pool (20) is deeper than the reported cutoffs (@1/@5/@10)** so the gold mapping isn't capped at k=10; the reviewer may also paste a raw `chunk_id` not in the pool. This *reduces* — does not eliminate — the bias of grading the retriever against gold that the same dense retriever surfaced.
  - **Unanswerable policy:** questions no docs chunk answers are marked `unanswerable` (`x` on the DECISION line) and **dropped from the scored set**, retained only as a **coverage stat** (correct for a docs-QA system per D-019). Blank decisions = `pending`, skipped with a warning.
  - **Metrics** (`src/eval/metrics.py`): Recall@k / Hit@k / nDCG@k at k=1,5,10 + MRR@3, binary relevance, macro-averaged — pure, LLM-free, unit-tested, deterministic (PLAN §6, CLAUDE.md §6). Baseline runner (`run_eval.py`) records config + git SHA + `judge_model=None` in every results file.
  - **High-confidence pre-fill** (`src/eval/prefill.py`, `./tasks.ps1 prefill`): the dense score can't separate a page's near-tied chunks (gap-to-#2 p75 ≈ 0.02), but when the **accepted answer explicitly links a docs page** that's in the candidate pool, that's a high-precision label. The tool URL-matches answer links against candidate `source_url` (normalizing away the `/oss/python|js/` language segment the rendered docs add but our chunk URLs omit), and pre-fills the **best chunk per linked page** (one representative — not every chunk of the page, which would inflate recall) into **blank** DECISION lines only, annotated for spot-check. Result on our set: **26/163** questions auto-suggested (20 with 1 gold chunk, 6 with 2). The reviewer confirms/edits; the remaining 137 are mapped by hand.
- **Why:** The batch flow makes ~163 hand-verifications a single offline pass; keeping metrics LLM-free makes the headline numbers maximally defensible; dropping unanswerables keeps the retrieval numbers honest while the drop-count itself is a reportable corpus-coverage figure.
- **Alternatives considered:** interactive review (slow); pure auto-map (biased, low credibility); pool = k=10 (would cap gold at the reported cutoff, inflating recall); micro-averaging metrics (lets multi-gold questions dominate — macro gives equal weight per question).
- **Tradeoffs / risks:** Residual dense-retriever bias in gold selection (mitigated by the deeper pool + manual-id escape hatch; disclosed in README). `review.md`/`candidates.jsonl` are large + regenerable → git-ignored; `gold.jsonl` is the committed durable artifact. Gold credibility still rests on the human pass actually being done carefully.
