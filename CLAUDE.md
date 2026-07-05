# CLAUDE.md — Operating Contract for This Project

> It is the source of truth for **how** to build. The **what** lives in `PROJECT_PLAN.md`.
> Every non-trivial choice is logged in `DECISIONS.md`. Session state lives in `PROGRESS.md`.

---

## 1. What we are building (one paragraph)

**DocsGPT-Agent** — a deployed, agentic, *cited* question-answering system over a fast-moving
developer-documentation corpus (LangChain/LangGraph docs + their GitHub Discussions/Issues, or
Kubernetes docs as the alternative). Unlike "chat with your PDF," it does **genuine multi-step
retrieval** (query decomposition → iterative retrieval → self-correction), **hybrid retrieval**
(dense + BM25 with RRF fusion), **cross-encoder reranking**, **per-claim citations**, and ships
with a **quantitative evaluation harness** (retrieval metrics + RAGAS) and **observability**
(Langfuse). It runs on **free tiers only** (GCP credits held as insurance).

The goal is a resume-defensible project for **both SDE and Data Science** interviews. That means:
strong engineering (deployed, observable, well-architected) **and** strong DS rigor (gold eval set,
ablation tables, honest hallucination measurement).

---

## 2. PRIME DIRECTIVE — build one layer at a time, then STOP

This is the single most important rule. **Violating it is a failure, even if the code is correct.**

1. Build **exactly one layer/component** from `PROJECT_PLAN.md` §"Build Layers" at a time.
2. When a layer is done, run its **Acceptance Gate** (the checklist for that layer) and show me the result.
3. Update `DECISIONS.md` (any non-trivial choice made during the layer) and `PROGRESS.md`.
4. **Then STOP and explicitly ask me for approval before starting the next layer.**
   Do not begin the next layer, do not "just scaffold ahead," do not batch two layers together.
5. If a layer turns out to be bigger than expected, split it and stop at the first sub-part.

When you finish a layer, end your message with:
`✅ Layer <N> complete. Gate results above. Shall I proceed to Layer <N+1>? (yes / adjust / stop)`

---

## 3. Decision-logging rule (for resume defense)

I must be able to explain **every non-trivial decision** in an interview. So:

- Whenever you make a choice that a reviewer could reasonably question — a library, a model, a
  chunk size, a fusion parameter, a metric threshold, a schema, a tradeoff — **append an entry to
  `DECISIONS.md`** using the template at the top of that file.
- Keep each entry short but complete: Context → Decision → Why → Alternatives considered → Tradeoffs/risks.
- If a decision reverses an earlier one, add a new entry that references the old one (don't silently edit history).
- Trivial choices (variable names, obvious formatting) do **not** need entries. Use judgment; when unsure, log it.

---

## 4. Hard constraints (do not violate without asking)

- **Free tier only.** No paid API calls, no paid cloud resources, without explicit approval.
  GCP free-trial credits exist as *insurance*, not the default. Every new dependency must have a free tier.
- **Deployable, not localhost.** The end state must run at a public URL (Cloud Run + Vercel + Qdrant Cloud).
- **Secrets never in git.** Use `.env` locally (git-ignored) and Secret Manager / env vars in deploy.
  Provide/maintain `.env.example` with keys but no values.
- **Reproducibility.** Pin dependency versions. The eval harness must produce the same numbers on re-run
  (fixed random seeds; fixed judge LLM for RAGAS — see PLAN §Eval).
- **Rate-limit awareness.** LLM calls must go through a single wrapper with exponential backoff + jitter on
  429s and model routing (see PLAN §LLM routing). Never fire unbounded parallel LLM calls.

---

## 5. Where things live

| File | Purpose |
|---|---|
| `CLAUDE.md` | This file — how to build (protocol, constraints, conventions). |
| `PROJECT_PLAN.md` | What to build — problem, corpus, stack, architecture, build layers, roadmap, metrics. |
| `DECISIONS.md` | Decision log with rationale. Update as you build. |
| `PROGRESS.md` | Running state: what's done, what's next, open questions, how to resume. Update every layer. |
| `.env.example` | Required environment variables (names only). |
| `README.md` | Built LAST (Layer 10) — the "paper": architecture diagram, ablation tables, honest caveats. |

At the **start of every session**: read `PROGRESS.md` first to see where we are, then continue from there.
At the **end of every layer**: update `PROGRESS.md` (done / next / blockers) so the next session can resume cleanly.

---

## 6. Coding conventions

- **Language:** Python 3.11+ for backend/ML; TypeScript/React for frontend.
- **Structure:** keep ingestion, retrieval, agent, eval, and API as separate importable modules
  (`src/ingest/`, `src/retrieval/`, `src/agent/`, `src/eval/`, `src/api/`). No monolithic script.
- **Config over constants:** all tunables (chunk size, top-k, RRF k, rerank top-n, model names) live in one
  `config.py` / `config.yaml`, not scattered as magic numbers. This makes ablations trivial and defensible.
- **Typed + documented:** type hints on public functions; a one-line docstring saying *why*, not just *what*.
- **Test the seams:** at minimum, a smoke test per layer that the Acceptance Gate can run.
- **Deterministic eval:** fixed seeds; log the exact model + params used for every eval run into the results file.
- **Small commits per layer:** one logical commit (or a few) per layer with a message referencing the layer number.

---

## 7. Interaction style I want from you

- Before writing code for a layer, give me a **2–4 line plan** of what you're about to do and any decision you're
  about to make that belongs in `DECISIONS.md`. If a decision is genuinely open, ask me rather than guessing.
- Prefer boring, well-supported libraries over clever ones. This is a portfolio project I must defend, not a playground.
- If something in `PROJECT_PLAN.md` looks wrong, outdated, or infeasible on free tier, **flag it and stop** —
  do not silently work around it. The plan may contain assumptions that need re-verification (esp. free-tier limits).
- Keep me in the loop on anything that spends GCP credits or approaches a free-tier limit.
