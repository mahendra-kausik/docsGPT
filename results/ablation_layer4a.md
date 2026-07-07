# Retrieval ablations — dense vs hybrid vs rerank vs decomposition

> **Default: HYBRID** (dense + BM25 + client-side RRF k=60). Rerank (D-030/D-032) and
> query decomposition (D-037) are **retained as selectable, measured pipelines** — not
> deleted — because both improve the **synthetic** slice and their numbers are part of
> the deliverable (D-039). They are *not the default* because both lose on the **real**
> forum slice, where plain dense/hybrid wins. Always read the synthetic gains together
> with the real-slice losses.
>
> Run any pipeline: `./tasks.ps1 eval --pipeline dense|hybrid|rerank|decomposed`.
>
> | recall@5 | dense | hybrid | rerank | decomposition |
> |---|---|---|---|---|
> | overall | 0.612 | **0.693** | 0.638 | 0.615 |
> | **forum (real)** | **0.577** | 0.404 | 0.327 | 0.173 |
> | synthetic | 0.622 | **0.768** | 0.718 | 0.730 |
>
> Takeaway: every technique beyond plain retrieval helps synthetic (lexically close to
> its gold) and hurts real (semantically matched) — an eval-validity finding, and the
> reason Layer 5's value is grounding honesty (D-038), not fancier retrieval.

## Three-way summary (n=126)

| Metric    | Dense | Hybrid | Rerank (MiniLM) | best     |
|-----------|-------|--------|-----------------|----------|
| recall@1  | 0.312 | 0.403  | 0.407           | rerank≈hybrid |
| recall@5  | 0.612 | **0.693** | 0.638        | **hybrid** |
| recall@10 | 0.734 | **0.778** | 0.698        | **hybrid** |
| mrr@3     | 0.511 | 0.614  | **0.616**       | rerank≈hybrid |
| ndcg@10   | 0.546 | **0.633** | 0.595        | **hybrid** |
| hit@5     | 0.738 | **0.802** | 0.754        | **hybrid** |
| latency p50 (ms) | 474 | 523 | 7306 (14×)   | dense    |

**Reranking (MiniLM):** helps rank-1 by a hair, degrades recall@5/@10/ndcg/hit, and
costs 14× latency. bge-reranker-base (possibly better ranking) measured ~25 s/query on
CPU — undeployable on free tier. Rejected (D-030). Real-slice numbers below are worse
still under rerank.

## Reranker bake-off — does ANY reranker beat hybrid? (Layer 4c, D-032)

ONNX rerankers (fastembed `TextCrossEncoder`) over the hybrid pool. Set: all 26 real
forum + 24 synthetic (n=50), pool 25, passages capped at `rerank_max_chars`. Latency is
rerank-only, local CPU fp32 — and it *is* the online per-query cost (batch = cost × N).
Source: `results/rerank_bakeoff_latest.json`.

| model | recall@5 | **forum (real)** | synth | mrr@3 | ndcg@10 | p50 (ms) |
|-------|----------|------------------|-------|-------|---------|----------|
| **hybrid (no rerank)** | **0.630** | **0.442** | 0.833 | 0.463 | 0.543 | **0** |
| ms-marco-MiniLM-L-6  | 0.550 | 0.288 | 0.833 | 0.410 | 0.485 | 2,816 |
| ms-marco-MiniLM-L-12 | 0.600 | 0.385 | 0.833 | 0.430 | 0.493 | 4,442 |
| BAAI/bge-reranker-base | 0.550 | 0.288 | 0.833 | 0.433 | 0.504 | 13,246 |
| jina-reranker-v2-multiling | 0.570 | 0.365 | 0.792 | **0.487** | **0.545** | 13,816 |

**No reranker beats hybrid on recall@5 or the real slice** — including the "strong"
bge (worst on the real slice *and* slowest, ~13 s/query at pool 25). But the split by
slice is the real story (out-of-domain hypothesis, confirmed):

- **Synthetic slice: rerankers HELP top-rank precision.** recall@5/@10 stay flat (gold
  already in the pool → reordering can't add recall), but recall@1 / mrr@3 / ndcg@10
  improve — jina-v2 lifts synth recall@1 0.583→**0.750**, ndcg 0.723→0.789; bge and
  MiniLM-L12 also help. Only weak MiniLM-L6 hurts. Here the query is lexically/
  semantically close to its gold passage — in-domain for the cross-encoder.
- **Real forum slice: rerankers HURT every metric.** Natural questions vs code-heavy
  chunks are out-of-distribution for cross-encoders trained on NL-query→prose, so they
  demote correct answers.

So it is not "rerankers are useless" — they sharpen the easy/in-domain questions and
damage the hard/real ones, and the real ones are what we ship for. Net recall@5 and the
real slice both favor hybrid; jina-v2 edges mrr@3/ndcg overall but loses recall@5 at
~14 s/query. **Reranking rejected on quality, not just latency (D-032).**

## The real-vs-synthetic split (why "hybrid wins" needs a caveat)

| recall@5  | Dense | Hybrid | Rerank |
|-----------|-------|--------|--------|
| **forum / real (n=26)** | **0.577** | 0.404 | 0.327 |
| synthetic (n=100)       | 0.622 | **0.768** | 0.718 |

On the **real** slice the order is **dense > hybrid > rerank** — the opposite of the
overall metric. The overall hybrid gain is entirely synthetic-driven; synthetic
questions were generated *from* their gold chunks so they share vocabulary BM25 rewards
(D-025/D-028). Honest reading: hybrid wins corpus-wide, but the small real slice favors
dense. Default = hybrid (standard, entity-robust); the real-slice weakness is left to
the Layer 5 agent (query decomposition / multi-query), not a reranker (D-031).

---

# Layer 4a ablation — dense-only vs hybrid (dense + BM25, client-side RRF k=60)

Same gold set (126 items: 26 real forum + 100 synthetic), same metrics, same Qdrant
cluster. Dense reuses `docs_dense`; hybrid uses `docs_hybrid` (dense vectors copied,
BM25 sparse added) fused client-side with RRF k=60 (D-027).

Source files:
- dense  → `results/eval_dense_20260706T173356Z.json`
- hybrid → `results/eval_hybrid_20260707T092021Z.json`

## Overall (n=126)

| Metric     | Dense | Hybrid | Δ        |
|------------|-------|--------|----------|
| recall@1   | 0.312 | 0.403  | **+0.091** |
| recall@5   | 0.612 | 0.693  | **+0.081** |
| recall@10  | 0.734 | 0.778  | +0.044   |
| mrr@3      | 0.511 | 0.614  | **+0.103** |
| ndcg@10    | 0.546 | 0.633  | +0.087   |
| hit@5      | 0.738 | 0.802  | +0.064   |
| latency p50 (ms) | 283 | 523 | +240 (2 round-trips + BM25; RTT-dominated) |

## Per source — the honest split (why the overall number is not the whole story)

| Metric    | Slice     | Dense | Hybrid | Δ        |
|-----------|-----------|-------|--------|----------|
| recall@5  | forum (real, n=26) | 0.577 | 0.404 | **−0.173** |
| mrr@3     | forum (real, n=26) | 0.404 | 0.250 | **−0.154** |
| recall@10 | forum (real, n=26) | 0.846 | 0.615 | **−0.231** |
| recall@5  | synthetic (n=100)  | 0.622 | 0.768 | +0.147   |
| mrr@3     | synthetic (n=100)  | 0.538 | 0.708 | +0.170   |
| recall@10 | synthetic (n=100)  | 0.705 | 0.820 | +0.115   |

## Reading

The overall +0.081 recall@5 is carried **entirely by the synthetic slice**; on the
real forum slice, hybrid **regressed** on every metric. Synthetic questions are
generated *from* their gold chunks, so they share surface vocabulary with the answer —
which BM25 rewards, flattering hybrid. Real forum questions match answers
**semantically, not lexically**; there, BM25's lexically-similar top hits dilute
dense's correct results through RRF and demote them (good dense hits fall out of the
top-10). This is a genuine effect, not a bug (sparse retrieval verified sensible;
direction consistent across all three real-slice metrics; n=26 is noisy).

**Implication:** this motivates (a) a k-sweep (k is tunable by design, D-027) and
(b) the Layer 4b cross-encoder reranker, whose job is to restore precision on the
fused candidates. The full dense-vs-hybrid-vs-rerank verdict lands after 4b.

## k-sweep — is the real-slice regression k-driven? (`results/sweep_rrf_*.json`)

Retrieval done once per query at the gate depth (10 from each list), fused at every k
(so the k=60 column reproduces the committed hybrid numbers exactly; only the merge
constant varies).

| metric / slice        | k=2   | k=10  | k=30  | k=60  | k=100 | k=200 |
|-----------------------|-------|-------|-------|-------|-------|-------|
| recall@5  overall     | 0.697 | 0.693 | 0.693 | 0.693 | 0.693 | 0.693 |
| mrr@3     overall     | 0.627 | 0.618 | 0.614 | 0.614 | 0.614 | 0.614 |
| recall@5  forum       | 0.442 | 0.404 | 0.404 | 0.404 | 0.404 | 0.404 |
| mrr@3     forum       | 0.295 | 0.269 | 0.250 | 0.250 | 0.250 | 0.250 |
| recall@10 forum       | 0.615 | 0.615 | 0.615 | 0.615 | 0.615 | 0.615 |
| recall@5  synthetic   | 0.763 | 0.768 | 0.768 | 0.768 | 0.768 | 0.768 |
| mrr@3     synthetic   | 0.713 | 0.708 | 0.708 | 0.708 | 0.708 | 0.708 |

**Verdict: the regression is NOT k-driven.** k barely moves any metric, and
**forum recall@10 is flat at 0.615 for every k** (dense was 0.846). The most
aggressive small-k (k=2) recovers forum recall@5 only to 0.442 — still well below
dense's 0.577. Reason: at depth 10, once BM25's top-10 contains lexically-similar-
but-wrong chunks for a natural-language query, the correct dense chunks fall **out of
the fused candidate set entirely** — no merge weighting can rank a chunk that is no
longer present. k=2 is marginally best overall (0.697) but within n=26 noise; not
worth changing the default from the standard k=60.

**Conclusion:** the fix is not the fusion constant — it is (1) a **deeper candidate
pool** so demoted-but-correct dense chunks stay available, plus (2) the **Layer 4b
cross-encoder reranker** to rescore them back to the top. 4b must rerank a deep fused
pool (~50), not the shallow fused top-10.
