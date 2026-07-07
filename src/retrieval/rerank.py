"""Cross-encoder reranking over the fused candidate pool (Layer 4b, D-007/D-029).

Why a cross-encoder here: bi-encoder retrieval (dense) and BM25 score query and
passage independently; a cross-encoder reads (query, passage) *together*, so it fixes
the precision problem the Layer 4a k-sweep exposed — hybrid demoted correct-but-not-
lexical chunks and no fusion constant could rerank an absent chunk (D-028). The fix is
to rerank a DEEP fused pool so those chunks are present as candidates, then let the
cross-encoder pull them back to the top. Model + pool depth come from config so this
is a one-line ablation (bge-reranker-base vs a lighter MiniLM if CPU latency is tight).
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

from src.config import get_settings
from src.retrieval.embedder import prepare_text

if TYPE_CHECKING:
    from src.retrieval.search import Hit


def apply_rerank(hits: list[Hit], scores: list[float], top_k: int) -> list[Hit]:
    """Reorder hits by cross-encoder score (desc), replacing each hit's score.

    Ties break by id so the eval is reproducible (CLAUDE.md §4). Pure + model-free so
    the reordering contract is unit-tested without loading torch.
    """
    ranked = sorted(zip(hits, scores, strict=True), key=lambda hs: (-hs[1], hs[0].id))
    return [replace(h, score=float(s)) for h, s in ranked[:top_k]]


@lru_cache(maxsize=2)
def _load_model(model_id: str):
    """Load a CrossEncoder once per id (cached). Imported lazily so the retrieval
    package stays importable (and unit-testable) without torch installed."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id)


class Reranker:
    """Scores (query, passage) pairs with a cross-encoder for precision reranking."""

    def __init__(self, model_id: str | None = None, max_chars: int | None = None) -> None:
        s = get_settings()
        self.model_id = model_id or s.reranker_model
        self.max_chars = max_chars or s.embed_max_chars

    @property
    def model(self):
        """The underlying (lazily loaded, cached) CrossEncoder."""
        return _load_model(self.model_id)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score per passage for the query (higher = more relevant)."""
        if not texts:
            return []
        # Same binary-scrub + length hygiene as the embedder, so the cross-encoder
        # sees clean passages (D-022); the query is short and passed as-is.
        pairs = [(query, prepare_text(t, self.max_chars)) for t in texts]
        return [float(s) for s in self.model.predict(pairs)]
