"""Retrieval metrics vs a gold set (Layer 3).

Why pure functions with no LLM: these are the most defensible numbers in the
project (PLAN §6) — Recall@k / MRR / nDCG / Hit Rate need only the ranked chunk
ids and the gold ids, so they reproduce exactly on re-run (CLAUDE.md §6). RAGAS
(judge-dependent) lives separately in Layer 3 Phase B.

Convention used throughout:
- ``ranked_ids``: the retriever's output, best-first, no duplicates.
- ``gold_ids``: the set of chunk ids that answer the question (>= 1 for a scored
  item). Relevance is binary — a chunk either answers the question or it doesn't.
Each function scores ONE query; :func:`aggregate` averages across queries.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _first_relevant_rank(ranked_ids: Sequence[str], gold: set[str], k: int) -> int | None:
    """1-based rank of the first gold hit within the top-k, or None if absent."""
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold:
            return rank
    return None


def hit_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """1.0 if any gold chunk appears in the top-k, else 0.0 (a.k.a. Hit Rate@k)."""
    gold = set(gold_ids)
    return 1.0 if _first_relevant_rank(ranked_ids, gold, k) is not None else 0.0


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """Fraction of the gold chunks that were retrieved within the top-k.

    With a single gold chunk (our common case) this collapses to Hit@k; it
    differs only when a question has multiple acceptable gold chunks.
    """
    gold = set(gold_ids)
    if not gold:
        return 0.0
    found = sum(1 for cid in ranked_ids[:k] if cid in gold)
    return found / len(gold)


def reciprocal_rank(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """1 / rank of the first gold hit within the top-k (0 if none) — MRR per query."""
    rank = _first_relevant_rank(ranked_ids, set(gold_ids), k)
    return 1.0 / rank if rank is not None else 0.0


def ndcg_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    DCG sums 1/log2(rank+1) over gold hits in the top-k; IDCG is the same for the
    ideal ranking (all gold packed at the top, capped at k). Normalizing makes the
    score comparable across questions with different numbers of gold chunks.
    """
    gold = set(gold_ids)
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, cid in enumerate(ranked_ids[:k], start=1) if cid in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


# The k values reported in the results file and README ablation table (PLAN §6).
DEFAULT_KS: tuple[int, ...] = (1, 5, 10)
# MRR is reported at a shallow cutoff — an answer buried past rank 3 is a miss.
MRR_K: int = 3


def score_query(
    ranked_ids: Sequence[str],
    gold_ids: Iterable[str],
    ks: Sequence[int] = DEFAULT_KS,
    mrr_k: int = MRR_K,
) -> dict[str, float]:
    """All per-query metrics as a flat dict, e.g. {'recall@5': 1.0, 'mrr@3': 0.5}."""
    gold = set(gold_ids)
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked_ids, gold, k)
        out[f"hit@{k}"] = hit_at_k(ranked_ids, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, gold, k)
    out[f"mrr@{mrr_k}"] = reciprocal_rank(ranked_ids, gold, mrr_k)
    return out


def aggregate(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    """Macro-average each metric across queries (equal weight per question)."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    n = len(per_query)
    return {key: sum(q[key] for q in per_query) / n for key in keys}
