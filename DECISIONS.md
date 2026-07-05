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
