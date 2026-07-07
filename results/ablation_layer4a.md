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
