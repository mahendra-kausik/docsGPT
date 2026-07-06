# PROGRESS.md — Running State

> **Claude Code: read this first at the start of every session; update it at the end of every layer.**
> Keep it short. This is how a new session resumes cleanly without re-reading everything.

## Current status
- **Active layer:** Layer 3 — Eval harness + gold set + baseline numbers (**retrieval baseline DONE**; RAGAS answer-quality = Phase B, pending)
- **Last completed layer:** Layer 2 — Indexing & dense baseline retrieval (gate passed 2026-07-06)
- **Layer 3 retrieval baseline (DONE, D-024/D-025):** gold set **rebalanced to synthetic-primary + real slice** after the forum-as-primary premise failed (114/163 answers link no docs). Final `data/gold/gold.jsonl` = **126 items** (26 real answer-link forum Qs + 100 synthetic: 70 single-hop / 30 multi-hop, Groq-8B-generated from chunks, gold-by-construction). New: LLM gateway (`src/llm/gateway.py`, Groq + backoff + cache), `synth.py`, `build_gold.py`; `run_eval.py` reports per-source. Tasks: `propose | prefill | synth | compile-gold | build-gold | eval`. **Dense-only baseline saved** to `results/eval_dense_20260706T173356Z.json`: overall recall@5 **0.612** / mrr@3 0.511 / recall@10 0.734 / ndcg@10 0.546; **real** recall@5 **0.577** / mrr@3 0.404 (n=26); synthetic recall@5 0.622 (n=100). This is the reference Layer 4 (hybrid+rerank) beats. 36 tests green; ruff clean.
- **Layer 3 Phase A (built, D-024):** eval harness under `src/eval/` — `metrics.py` (Recall@k/Hit@k/nDCG@k @1,5,10 + MRR@3, pure/LLM-free, unit-tested), `gold.py` (GoldItem + forum accepted-answer extraction), `propose.py` (+`--from-candidates` offline re-render), `prefill.py`, `compile_gold.py`, `run_eval.py`. Tasks: `./tasks.ps1 propose | prefill | compile-gold | eval`. **33 tests green; ruff clean.** Fixed a Windows cp1252 console crash (non-ASCII in prints). Pipeline smoke-tested end-to-end (compile→eval ran clean).
  - **Generated** `data/gold/review.md` (163 Qs, each w/ accepted forum answer + 20 dense candidates) + `data/gold/candidates.jsonl` (both git-ignored, regenerable).
  - **Pre-filled 26/163** DECISION lines via `prefill` (answer-link URL match, best chunk per linked page; annotated for spot-check). **137 still blank — need manual mapping.**
  - **NEXT (user):** verify the 26 auto-suggested + map the 137 blank DECISION lines in `review.md` → `./tasks.ps1 compile-gold` (writes durable `data/gold/gold.jsonl`) → `./tasks.ps1 eval` (saves dense baseline to `results/eval_dense_*.json` = Layer 3 gate). Preliminary eval on the 26 auto-labeled only: recall@1 .27 / @5 .58 / @10 .85, mrr@3 .40 (biased subset, NOT the baseline). Groq/RAGAS = Phase B.
- **Build order & gates:** see `PROJECT_PLAN.md` §4. (Layer 1 split into 1a docs / 1b labels — see D-016; label source pivoted to the Forum — see D-018/D-019.)
- **Env:** Windows 11 / Windows PowerShell 5.1; Python 3.13.3; `.venv` at repo root. Run tasks with `./tasks.ps1 <setup|test|lint|format|ingest|ingest-forum>`.
- **Repo:** git initialized; remote `origin` = https://github.com/mahendra-kausik/docsGPT.git. Commits authored by user only (no Claude co-author).
- **Corpus (1a):** 11,035 chunks / 751 files from MIT `langchain-ai/docs` @ sha 662d399 → `data/corpus/chunks.jsonl` (+ `manifest.json`). Rebuild: `./tasks.ps1 ingest`. Raw clone in `data/raw/` (git-ignored).
- **Gold seeds (1b):** 163 solved-topic question→accepted-answer pairs from the LangChain Forum OSS category → `data/gold/forum_seeds.jsonl` (question + metadata + answer URL; **no answer body**, D-019). Refetch: `./tasks.ps1 ingest-forum`. Full answers cached in `data/raw/forum/` (git-ignored) for Layer 3 mapping.
- **Qdrant (Layer 2 infra):** free cluster **live** on GCP / `us-central1` (co-located with Cloud Run); endpoint `https://…us-central1-0.gcp.cloud.qdrant.io:6333`, auth-verified (200 OK). Creds in `.env` (`QDRANT_URL` + `QDRANT_API_KEY`). ⚠️ free clusters suspend after 1 wk idle / delete after 4 wk — need a keep-alive (Layer 8, D-004).
- **Dense index (2):** all **11,035 chunks** embedded with `bge-small-en-v1.5` (384-dim, cosine, normalized) → Qdrant collection **`docs_dense`** (`collection_count=11035`). Modules: `src/retrieval/{embedder,index,search}.py`. Rebuild: `./tasks.ps1 index` (~30 min, one-time; slow due to `wait=True` upserts on free tier). Query: `./tasks.ps1 search "..."`. **Gate:** top-k on 5 real queries all on-target (scores 0.77–0.86); **warm p50 ≈ 283 ms** measured India→US (mostly network RTT; single-digit ms expected from co-located Cloud Run); cold model load ~13 s once/process.

## Layer checklist
- [x] Layer 0 — Repo scaffold & config
- [x] Layer 1a — Docs ingestion & chunking
- [x] Layer 1b — Forum gold-eval seeds (natural labels for Layer 3)
- [x] Layer 2 — Indexing & dense baseline retrieval
- [ ] Layer 3 — Eval harness + gold set + baseline numbers  ← do not skip
- [ ] Layer 4 — Hybrid retrieval + reranker (ablation table)
- [ ] Layer 5 — Agentic loop + citations
- [ ] Layer 6 — API + streaming + rate-limit hardening
- [ ] Layer 7 — Observability
- [ ] Layer 8 — Deployment (public URL)
- [ ] Layer 9 — Product skin (optional)
- [ ] Layer 10 — Polish & defense

## Decisions log
- Pre-seeded D-001…D-012 in `DECISIONS.md`. Layer 0: D-013 (PowerShell task runner), D-014 (pyproject + pinned requirements), D-015 (lean deps + pydantic-settings). Layer 1a: D-016 (MIT docs-repo source, Python-focused scope; licensing verified), D-017 (hand-rolled fence-aware MDX cleaning + chunking). Layer 1b: D-018 (labels from LangChain Forum, not GitHub Discussions — migrated), D-019 (forum data = gold-eval seeds only, not corpus; leakage + licensing). Layer 2 prep: D-020 (free-tier limits re-verified 2026-07-06; `gemini-2.0-flash` retired → `gemini-2.5-flash`; Groq TPD now binding). Layer 2: D-021 (Qdrant cosine/normalized, model-derived dim, full payload, uuid5 point ids, bge query instruction), D-022 (base64/data-URI scrub before embed), D-023 (torch CPU build; CPU wheel index at Layer 8). Layer 3: D-024 (gold build = batch propose→review.md→compile; drop docs-unanswerable Qs; pure LLM-free retrieval metrics).

## Open questions / blockers
- ✅ Free-tier limits re-verified 2026-07-06 (D-020): Qdrant, Cloud Run, Groq, Gemini done. **Still to verify at their layers:** Vercel Hobby + Langfuse cap (§7).
- **Layer 3:** map the 163 forum questions → MIT docs chunk(s) that answer them (read answers from `data/raw/forum/`), hand-verify, augment with RAGAS synthetic to reach the gold target. Drop questions not answerable from docs (corpus gaps). Fix RAGAS judge (Groq 8B).
- ✅ Layer 2 wheels verified (Python 3.13 / Windows): `torch 2.12.1` (cp313 win_amd64), `sentence-transformers 5.6.0` / `transformers 5.13.0` / `qdrant-client 1.18.0` (pure-python). Pin these when added.
- ✅ Layer 2 embedding hygiene handled (D-022): `scrub_binary()` strips base64/data-URI runs before embed; length capped at `embed_max_chars`. Unit-tested.

## How to resume
1. Read this file, then `CLAUDE.md`, then the relevant section of `PROJECT_PLAN.md`.
2. Continue from the active layer. Build only that layer, run its gate, update this file + `DECISIONS.md`, then STOP and ask.
