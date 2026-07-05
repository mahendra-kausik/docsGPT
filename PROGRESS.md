# PROGRESS.md — Running State

> **Claude Code: read this first at the start of every session; update it at the end of every layer.**
> Keep it short. This is how a new session resumes cleanly without re-reading everything.

## Current status
- **Active layer:** Layer 1 — Corpus ingestion & chunking (NOT STARTED)
- **Last completed layer:** Layer 0 — Repo scaffold & config (gate passed 2026-07-05)
- **Build order & gates:** see `PROJECT_PLAN.md` §4.
- **Env:** Windows 11 / Windows PowerShell 5.1; Python 3.13.3; `.venv` at repo root. Run tasks with `./tasks.ps1 <setup|test|lint|format>`.
- **Repo:** git initialized; remote `origin` = https://github.com/mahendra-kausik/docsGPT.git. Commits authored by user only (no Claude co-author).

## Layer checklist
- [x] Layer 0 — Repo scaffold & config
- [ ] Layer 1 — Corpus ingestion & chunking
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
- Pre-seeded D-001…D-012 in `DECISIONS.md`. Layer 0 added D-013 (PowerShell task runner), D-014 (pyproject + pinned requirements), D-015 (lean layer-incremental deps + pydantic-settings config).

## Open questions / blockers
- ⚠️ Re-verify free-tier limits before relying on them (Gemini RPM/RPD in AI Studio; Groq per-model RPD; Qdrant inactivity windows; Cloud Run quotas/regions; Vercel Hobby; Langfuse cap). See `PROJECT_PLAN.md` §7.
- **Layer 1 first task:** confirm final corpus + its docs licensing before scraping (LangChain/LangGraph docs + GitHub, per D-002); check each source's LICENSE/site terms.
- ⚠️ At Layer 2: verify torch / sentence-transformers ship Python 3.13 Windows wheels before adding them.

## How to resume
1. Read this file, then `CLAUDE.md`, then the relevant section of `PROJECT_PLAN.md`.
2. Continue from the active layer. Build only that layer, run its gate, update this file + `DECISIONS.md`, then STOP and ask.
