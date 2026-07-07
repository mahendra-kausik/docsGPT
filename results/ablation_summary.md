# Retrieval ablation — best config per slice (headline numbers)

One-page summary of the retrieval ablation for the README / resume. All numbers are
reproducible: `./tasks.ps1 eval --pipeline dense|hybrid|rerank|decomposed` over the
committed gold set. Retrieval metrics are pure (no LLM judge) and macro-averaged.

**Gold set (126):** 26 real questions from the LangChain Forum (accepted-answer →
docs-page labels) + 100 synthetic (Groq-generated from docs chunks: 70 single-hop,
30 multi-hop). Reported **per source** so the real vs synthetic gap is never hidden.

**Pipelines:** dense (bge-small) · hybrid (dense + BM25, client-side RRF k=60) ·
rerank (hybrid → MiniLM cross-encoder over a 50-pool) · decomposed (Groq query
decomposition → multi-query hybrid → RRF).

---

## Best configuration per slice

| Slice | Best pipeline | recall@5 | recall@1 | mrr@3 | ndcg@10 | note |
|-------|---------------|----------|----------|-------|---------|------|
| **Synthetic (n=100)** | **hybrid** | **0.768** | 0.473 | 0.708 | **0.704** | +0.146 recall@5 over dense |
| Synthetic — top-rank | rerank | 0.718 | **0.483** | **0.727** | 0.687 | rerank wins r@1/mrr@3 only |
| **Real forum (n=26)** | **dense** | **0.577** | **0.269** | **0.404** | **0.524** | every added module regresses it |

**Two headline results, both honest:**

1. **On synthetic questions, hybrid retrieval clearly beats the dense baseline** —
   recall@5 **0.622 → 0.768 (+24%)**, mrr@3 0.538 → 0.708, ndcg@10 0.552 → 0.704.
   A cross-encoder reranker adds a marginal top-rank gain (recall@1 0.473 → 0.483,
   mrr@3 0.708 → 0.727) at ~14× latency.
2. **On real user questions, plain dense retrieval is strongest** — recall@5 **0.577**,
   and *every* added module (BM25 fusion, reranking, decomposition) regresses it.

---

## Full matrix

### Synthetic (n=100) — hybrid wins recall, rerank wins top-precision
| pipeline | recall@1 | recall@5 | recall@10 | mrr@3 | ndcg@10 | p50 ms |
|----------|----------|----------|-----------|-------|---------|--------|
| dense | 0.323 | 0.622 | 0.705 | 0.538 | 0.552 | 474 |
| **hybrid** | 0.473 | **0.768** | **0.820** | 0.708 | **0.704** | 523 |
| rerank | **0.483** | 0.718 | 0.785 | **0.727** | 0.687 | 7306 |
| decomposed | 0.407 | 0.730 | 0.807 | 0.668 | 0.664 | 2415 |

### Real forum (n=26) — dense wins every metric
| pipeline | recall@1 | recall@5 | recall@10 | mrr@3 | ndcg@10 | p50 ms |
|----------|----------|----------|-----------|-------|---------|--------|
| **dense** | **0.269** | **0.577** | **0.846** | **0.404** | **0.524** | 474 |
| hybrid | 0.135 | 0.404 | 0.615 | 0.250 | 0.360 | 523 |
| rerank | 0.115 | 0.327 | 0.365 | 0.192 | 0.237 | 7306 |
| decomposed | 0.135 | 0.173 | 0.327 | 0.173 | 0.211 | 2415 |

### Overall (n=126, mixed) — hybrid best on aggregate
| pipeline | recall@1 | recall@5 | recall@10 | mrr@3 | ndcg@10 | p50 ms |
|----------|----------|----------|-----------|-------|---------|--------|
| dense | 0.312 | 0.612 | 0.734 | 0.511 | 0.546 | 474 |
| **hybrid** | 0.403 | **0.693** | **0.778** | 0.614 | **0.633** | 523 |
| rerank | 0.407 | 0.638 | 0.698 | 0.616 | 0.595 | 7306 |
| decomposed | 0.350 | 0.615 | 0.708 | 0.566 | 0.570 | 2415 |

---

## The finding worth talking about

The two slices **disagree about what "better retrieval" means**, and that disagreement
is the most defensible result here:

- **Synthetic questions were generated *from* their gold chunks**, so they share surface
  vocabulary with the answer. BM25 (in hybrid) and cross-encoders reward that overlap →
  they score well on synthetic.
- **Real questions are natural language, matched to answers *semantically not lexically***.
  Adding keyword signal (BM25), re-ranking with a web-trained cross-encoder, or fusing
  query variants all pull retrieval toward lexical matches and **demote the correct
  semantic hit** — so they regress the real slice.

**Takeaway:** the synthetic slice systematically *overstates* the value of retrieval
"sophistication." Measuring per-slice against real labels revealed that a dense baseline
is the strongest retriever for real user questions on this corpus (n=26 — small, so
directional). This is why the deployed agent's value comes from **grounding honesty**
(answer only from retrieved context, verify, and refuse — see the agent layer), not from
squeezing more retrieval recall.

*Default pipeline shipped: hybrid (best on aggregate + synthetic). All four pipelines are
retained and runnable so any slice can be re-measured with any configuration.*
