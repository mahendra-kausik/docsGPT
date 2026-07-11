# DocsGPT-Agent

A deployed, agentic, **cited** question-answering system over the LangChain/LangGraph developer
docs (11,035 chunks, 751 pages). Hybrid retrieval, LangGraph self-correction, per-claim citations,
a quantitative eval harness (retrieval metrics + RAGAS), and Langfuse tracing — running entirely on
free-tier infrastructure.

**Live demo:** https://frontend-three-gamma-49.vercel.app
**API:** https://docsgpt-agent-ee54gmitka-uc.a.run.app (`/docs` for OpenAPI, `/ask`, `/ask/stream`)

> Built to be resume-defensible for both SDE and Data Science interviews: every non-trivial
> engineering choice and every headline number below traces to a decision in `DECISIONS.md`
> (52 entries) or a results file under `results/`. Where an idea was tried and **rejected**
> (reranking, query decomposition), that's reported too — the honest negative results are as
> load-bearing here as the positive ones.

---

## Why this is a harder problem than "chat with your PDF"

Developer docs for a fast-moving framework are a poor naive-RAG target: real questions
("how do I stream tokens from a chain?") are entity/token-heavy (exact API names, args) and are
answered *semantically*, not lexically — while synthetic/generated eval questions tend to share
vocabulary with their source chunk and reward lexical tricks. This gap turned out to be the single
most important empirical finding in the project (see the ablation below): several standard
"retrieval sophistication" upgrades measurably **helped synthetic questions and hurt real ones**.
The project's actual defensibility comes from measuring that gap honestly, not hiding it, and
shipping an agent that would rather refuse than hallucinate.

---

## Architecture

```
User ──HTTPS/SSE──► Vercel (React + TS, Vite)         "frontend/"
                       │
                       ▼
             Cloud Run (FastAPI, scale-to-zero)        "src/api/"
                       │
                       ▼
          LangGraph agent  retrieve → synthesize → verify
                              │            │          │
                              │            │          └─ ungrounded? ─► retry (bounded) ─► synthesize
                              │            │                        └─► refuse ─► cite
                              │            ▼
                              │       LLM Gateway (single wrapper: 429 backoff+jitter, cache, routing)
                              │        ├─ Groq llama-3.3-70b-versatile  (deployed default synthesis)
                              │        ├─ Groq llama-3.1-8b-instant     (verify / cheap nodes / RAGAS judge)
                              │        └─ Gemini 2.5 Flash              (opt-in per request, 20 req/day cap)
                              ▼
                     Qdrant Cloud (free tier)
                       dense (bge-small-en-v1.5) + BM25 sparse, client-side RRF fusion, k=60
                       └─ docs_hybrid (deployed default) / docs_dense (baseline, kept for ablation)

  Langfuse Cloud: every LangGraph node = a trace span; every real LLM call = a generation span
                  (tokens, latency); optional/no-op with no API key so tests/CLI stay hermetic.
```

**Agent flow** (the differentiator): `retrieve → synthesize → verify(grounding) → {cite | retry→resynthesize | refuse→cite}`.
Two ideas from the original design were built, measured, and **rejected**:
- **Query decomposition / multi-query retrieval** — hurt recall on real questions (see below); dropped.
- **Pre-synthesis relevance grading** — the grader inherited the LLM's own prior ("I already know the
  capital of France") and couldn't tell "I know this" from "the passages say this." Replaced with
  **post-synthesis grounding verification**: draft an answer, then have a cheap model check whether the
  *draft's claims* are actually supported by the cited passages. This is what makes "capital of
  France?" (out of corpus) correctly produce a refusal instead of a confidently wrong, citation-decorated
  answer.
- **Self-correction** re-synthesizes over the *same* retrieved context with corrective feedback, not a
  fresh retrieval — re-retrieval was already shown to hurt real questions, so a retry that re-queries
  would just repeat that failure mode.

---

## Retrieval ablation — the headline result

Gold set: **126 questions** — 26 real (LangChain Forum accepted-answer → docs-page, hand-verified) +
100 synthetic (Groq-8B generated from chunks, gold-by-construction). Reported **per source**, never
blended, because the two slices disagree about what "better retrieval" means.

| Slice | Best pipeline | recall@5 | mrr@3 | ndcg@10 |
|---|---|---|---|---|
| **Synthetic (n=100)** | hybrid | **0.768** (+24% vs dense) | 0.708 | 0.704 |
| **Real forum (n=26)** | **dense** | **0.577** | **0.404** | **0.524** |
| Overall (n=126) | hybrid | 0.693 | 0.616 | 0.634 |

Full matrix (`results/ablation_summary.md`, `results/eval_{dense,hybrid,rerank,decomposed}_*.json`):

| pipeline | overall recall@5 | real recall@5 (n=26) | synthetic recall@5 (n=100) | p50 latency |
|---|---|---|---|---|
| dense (baseline) | 0.612 | **0.577** | 0.622 | 474 ms |
| **hybrid (BM25+RRF) — shipped default** | **0.693** | 0.404 | **0.768** | 523 ms |
| + cross-encoder rerank | 0.638 | 0.327 | 0.718 | 7,306 ms (14×) |
| + query decomposition | 0.615 | 0.173 | 0.730 | 2,415 ms (5×) |

**Why hybrid ships as the default despite losing on the real slice:** it wins on aggregate and on the
larger synthetic slice, and the real-forum weakness is a retrieval-*ranking* problem the agent's
grounding/refuse loop already defends against (a wrong-context answer gets caught by verification,
not silently shipped). Reranking and query decomposition were rejected outright — both cost
5–14× latency for a net loss on the slice that matters most (real questions). All four pipelines are
still runnable (`--pipeline dense|hybrid|rerank|decomposed`) as documented ablations, not deleted.

**Root cause:** synthetic questions are generated *from* their gold chunk, so they share surface
vocabulary with the answer — BM25 and cross-encoders reward that overlap. Real questions are natural
language matched *semantically*; adding lexical signal demotes the correct semantic hit. Measuring
per-slice against real labels is what surfaced this — an aggregate-only number would have hidden it
and shipped a regression against real users.

---

## Answer-quality (RAGAS) — measured, not idealized

Judge fixed to Groq `llama-3.1-8b-instant` throughout (routed through the project's own LLM gateway,
not `langchain-groq`, so caching/backoff/routing stay uniform). n=12 (6 forum + 6 synthetic, seed=13)
— small by design to stay inside Groq's free-tier TPM cap; treat as directional, not statistically tight.

| metric | deployed default (Groq 70B synthesis) | earlier run (Groq 8B synthesis) |
|---|---|---|
| faithfulness | 0.361 (n=11) | 0.311 (n=9) |
| answer_relevancy | 0.228 (n=12) | 0.380 (n=12) |
| answer_correctness | 0.381 (n=4) | 0.404 (n=4) |
| context_recall | 0.449 (n=6) | 0.449 (n=6) — identical, retrieval-only metric |
| context_precision | 0.333 (n=6) | 0.333 (n=6) — identical, retrieval-only metric |

context_recall/precision are byte-identical across runs because they only depend on retrieved
context, not the synthesis model — a useful internal-consistency check that the harness is
measuring what it claims to. On synthesis-side metrics, going from 8B→70B raised faithfulness but
lowered relevancy; n=12 is too small to call this a trend rather than noise, so both numbers are
reported rather than the more flattering one (`DECISIONS.md` D-052).

**These are not great absolute numbers, on purpose reported as such.** Two known depressants: (1)
the RAGAS judge context is capped at 4 chunks × 600 chars to stay under Groq's 6,000 TPM free cap,
which biases faithfulness down versus judging the full 8-chunk context the agent actually sees; (2)
free-tier judge calls occasionally fail (413/parse errors) — dropped from the mean rather than
scored zero, so `n` is reported per-metric alongside the mean.

---

## Hallucination handling — the actual safety story

Grounding is enforced structurally, not just measured: a post-synthesis verifier (Groq 8B) checks
whether the drafted answer's claims are supported by its cited passages. If not: one bounded
self-correction retry (re-synthesize over the same context with corrective feedback), then refusal
("I don't know based on the provided documentation") rather than shipping an unsupported answer.
Measured case: on a weak-retrieval real question, Gemini 2.5 Flash correctly refused while Groq 70B
initially confabulated a plausible-looking API call that the 8B verifier caught, triggering a retry
that produced a grounded answer instead (`DECISIONS.md` D-046/D-050). Citations are resolved and
validated post-hoc — invalid citations are surfaced in the response (`invalid_citations`), never
silently dropped.

---

## Engineering

- **Deployment:** FastAPI + LangGraph agent on Cloud Run (scale-to-zero, `--allow-unauthenticated`),
  React/TS (Vite) on Vercel. Both models baked into the Cloud Run image (no cold-start download from
  HF past the request timeout). CORS locked to an explicit origin allow-list.
- **Streaming:** SSE (`fetch()` + `ReadableStream`, not `EventSource` — the endpoint is POST) streams
  lifecycle stage events (retrieve/synthesize/verify/retry/refuse) then the *verified* answer
  token-by-token — never the raw pre-verification draft, since verification can retract it.
- **Rate-limit hardening:** one LLM gateway wraps every Groq/Gemini call with exponential
  backoff+jitter and a response cache; a simulated-429 recovery path is unit-tested.
- **Durability:** the Qdrant free cluster auto-deletes after ~4 weeks idle. A one-command `reindex`
  task rebuilds both collections from the committed `data/corpus/chunks.jsonl` (source of truth,
  never re-scraped); a weekly Cloud Scheduler job pings `/ping` to keep the cluster alive.
- **Observability:** every LangGraph node and every real LLM call ships a Langfuse trace span
  (tokens, latency); optional/no-op with no API key so tests, the CLI, and eval runs need no account.
- **Latency (hybrid, live):** p50 579 ms / p95 692 ms end-to-end retrieval; full agent turn (retrieval
  + synthesis + verify) 2–4 s warm on Cloud Run.

---

## Honest caveats (read before quoting a number from this README)

1. **Real-question retrieval (n=26) is small.** Directional, not conclusive — but it's the only slice
   with genuinely independent natural labels (not generated by the same model being evaluated).
2. **Hybrid is shipped as default despite losing the real slice** — an explicit, logged tradeoff
   (`DECISIONS.md` D-028/D-031), not an oversight.
3. **RAGAS numbers are judge-context-capped** to fit Groq's free-tier TPM limit, biasing faithfulness
   down relative to the agent's actual 8-chunk context.
4. **The 70B-vs-8B faithfulness/relevancy drift (D-052) is measured on n=12** and could be noise, not
   a confirmed trend — reported both ways rather than picking the better-looking number.
5. **Free tier only, by design** — Groq/Gemini/Qdrant/Cloud Run/Vercel/Langfuse free tiers, no paid
   calls. This bounds throughput (Gemini: 20 req/day) and cluster durability (Qdrant: idle-delete),
   both mitigated but not eliminated.

---

## Tech stack

Python 3.11+ (backend/agent/eval) · React + TypeScript/Vite (frontend) · LangGraph (agent
orchestration) · Qdrant Cloud (dense + BM25 sparse, client-side RRF) · BAAI bge-small-en-v1.5
(embeddings) · Groq (Llama 3.1 8B / 3.3 70B) + Gemini 2.5 Flash (LLMs, routed through one gateway) ·
`ragas==0.4.3` (answer-quality eval) · Langfuse (observability) · FastAPI + SSE · Docker on Cloud Run
· Vercel.

## Reproducing the numbers in this README

```
./tasks.ps1 eval --pipeline hybrid      # retrieval metrics -> results/eval_hybrid_<ts>.json
./tasks.ps1 ragas --sample 12 --seed 13 # RAGAS answer-quality -> results/ragas_<ts>.json
```
Every results file records its exact config, git SHA, and judge/synthesis model, so any number here
can be traced back to a specific run.

## Repo layout

```
src/ingest/     corpus fetch + structure-aware chunking
src/retrieval/  embedding, dense/hybrid/rerank retrievers, RRF fusion
src/agent/      LangGraph state machine: retrieve/synthesize/verify/retry/refuse, citations
src/eval/       gold-set build, retrieval metrics, RAGAS harness
src/api/        FastAPI app (JSON + SSE)
src/llm/        provider-aware LLM gateway (routing, backoff, cache)
src/obs/        Langfuse tracing wrapper (optional/no-op)
frontend/       React/TS/Vite UI
results/        every eval run's results JSON (config + git SHA + judge recorded)
DECISIONS.md    the decision log — every non-trivial choice, why, alternatives, tradeoffs
PROGRESS.md     session-by-session build log
```
