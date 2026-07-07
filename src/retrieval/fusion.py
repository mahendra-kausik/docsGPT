"""Client-side Reciprocal Rank Fusion for hybrid retrieval (Layer 4a, D-027).

Why client-side rather than Qdrant's server-side FusionQuery: qdrant-client 1.18's
FusionQuery exposes no `k`, so the RRF rank constant can't be set — it silently uses
Qdrant's internal value. Fusing here honours the documented k=60 (Cormack et al. 2009)
AND keeps `k` a real, sweepable knob (rrf_k in config) so the dense-vs-hybrid table can
report a k-ablation. RRF needs no score-scale normalisation, which is its whole appeal
over weighted fusion (D-006).
"""

from __future__ import annotations


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id-lists by RRF: score(d) = sum 1/(k + rank_i(d)), rank 1-based.

    Returns (id, score) pairs sorted by score descending, id ascending on ties so the
    output is deterministic (a re-run must reproduce the eval numbers — CLAUDE.md §4).
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
