# PROGRESS.md — Running State

> **Claude Code: read this first at the start of every session; update it at the end of every layer.**
> Keep it short. This is how a new session resumes cleanly without re-reading everything.

## Current status
- **Active layer:** Layer 2 — Indexing & dense baseline retrieval (NOT STARTED)
- **Last completed layer:** Layer 1b — Forum gold-eval seeds (gate passed 2026-07-06)
- **Build order & gates:** see `PROJECT_PLAN.md` §4. (Layer 1 split into 1a docs / 1b labels — see D-016; label source pivoted to the Forum — see D-018/D-019.)
- **Env:** Windows 11 / Windows PowerShell 5.1; Python 3.13.3; `.venv` at repo root. Run tasks with `./tasks.ps1 <setup|test|lint|format|ingest|ingest-forum>`.
- **Repo:** git initialized; remote `origin` = https://github.com/mahendra-kausik/docsGPT.git. Commits authored by user only (no Claude co-author).
- **Corpus (1a):** 11,035 chunks / 751 files from MIT `langchain-ai/docs` @ sha 662d399 → `data/corpus/chunks.jsonl` (+ `manifest.json`). Rebuild: `./tasks.ps1 ingest`. Raw clone in `data/raw/` (git-ignored).
- **Gold seeds (1b):** 163 solved-topic question→accepted-answer pairs from the LangChain Forum OSS category → `data/gold/forum_seeds.jsonl` (question + metadata + answer URL; **no answer body**, D-019). Refetch: `./tasks.ps1 ingest-forum`. Full answers cached in `data/raw/forum/` (git-ignored) for Layer 3 mapping.
- **Qdrant (Layer 2 infra):** free cluster **live** on GCP / `us-central1` (co-located with Cloud Run); endpoint `https://…us-central1-0.gcp.cloud.qdrant.io:6333`, auth-verified (200 OK). Creds in `.env` (`QDRANT_URL` + `QDRANT_API_KEY`). ⚠️ free clusters suspend after 1 wk idle / delete after 4 wk — need a keep-alive (Layer 8, D-004).

## Layer checklist
- [x] Layer 0 — Repo scaffold & config
- [x] Layer 1a — Docs ingestion & chunking
- [x] Layer 1b — Forum gold-eval seeds (natural labels for Layer 3)
- [ ] Layer 2 — Indexing & dense baseline retrieval
- [ ] Layer 3 — Eval harness + gold set + baseline numbers  ← do not skip
- [ ] Layer 4 — Hybrid retrieval + reranker (ablation table)
- [ ] Layer 5 — Agentic loop + citations
- [ ] Layer 6 — API + streaming + rate-limit hardening
- [ ] Layer 7 — Observability
- [ ] Layer 8 — Deployment (public URL)
- [ ] Layer 9 — Product skin (optional)
- [ ] Layer 10 — Polish & defense

## Decisions log
- Pre-seeded D-001…D-012 in `DECISIONS.md`. Layer 0: D-013 (PowerShell task runner), D-014 (pyproject + pinned requirements), D-015 (lean deps + pydantic-settings). Layer 1a: D-016 (MIT docs-repo source, Python-focused scope; licensing verified), D-017 (hand-rolled fence-aware MDX cleaning + chunking). Layer 1b: D-018 (labels from LangChain Forum, not GitHub Discussions — migrated), D-019 (forum data = gold-eval seeds only, not corpus; leakage + licensing). Layer 2 prep: D-020 (free-tier limits re-verified 2026-07-06; `gemini-2.0-flash` retired → `gemini-2.5-flash`; Groq TPD now binding).

## Open questions / blockers
- ✅ Free-tier limits re-verified 2026-07-06 (D-020): Qdrant, Cloud Run, Groq, Gemini done. **Still to verify at their layers:** Vercel Hobby + Langfuse cap (§7).
- **Layer 3:** map the 163 forum questions → MIT docs chunk(s) that answer them (read answers from `data/raw/forum/`), hand-verify, augment with RAGAS synthetic to reach the gold target. Drop questions not answerable from docs (corpus gaps). Fix RAGAS judge (Groq 8B).
- ✅ Layer 2 wheels verified (Python 3.13 / Windows): `torch 2.12.1` (cp313 win_amd64), `sentence-transformers 5.6.0` / `transformers 5.13.0` / `qdrant-client 1.18.0` (pure-python). Pin these when added.
- ⚠️ Layer 2 embedding hygiene: 2 chunks are ~52k chars dominated by a ~52k base64 run; ~3 more have base64 runs 600–1,100 chars. **Scrub base64/data-URI runs before embedding**; tokenizer truncates the genuinely-long code chunks (corpus median 561 / p95 1365 chars).

## How to resume
1. Read this file, then `CLAUDE.md`, then the relevant section of `PROJECT_PLAN.md`.
2. Continue from the active layer. Build only that layer, run its gate, update this file + `DECISIONS.md`, then STOP and ask.
