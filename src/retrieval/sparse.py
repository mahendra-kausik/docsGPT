"""BM25 sparse embedding for hybrid retrieval (Layer 4a, D-026).

Why fastembed's "Qdrant/bm25" rather than a self-rolled BM25: the dense-vs-hybrid
delta is the headline ablation, so the sparse half must be correct — a subtle
in-house BM25 bug would quietly weaken hybrid and mislead the table (D-026). BM25's
IDF term is applied server-side via the sparse vector's Modifier.IDF, so documents and
queries carry only the term-frequency component computed here. The same input hygiene
as the dense path (scrub base64/data-URIs, D-022) keeps the two vector spaces aligned.
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import models

from src.config import get_settings
from src.retrieval.embedder import prepare_text


@lru_cache(maxsize=2)
def _load_model(model_id: str):
    """Load a SparseTextEmbedding once per id (cached). Imported lazily so the rest
    of the retrieval package stays importable without fastembed/onnxruntime."""
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_id)


def _to_sparse_vector(embedding) -> models.SparseVector:
    """Convert a fastembed SparseEmbedding to a Qdrant SparseVector."""
    return models.SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )


class SparseEmbedder:
    """Encodes text into BM25 sparse vectors (term-frequency component; IDF server-side)."""

    def __init__(self, model_id: str | None = None, max_chars: int | None = None) -> None:
        s = get_settings()
        self.model_id = model_id or s.sparse_model
        self.max_chars = max_chars or s.embed_max_chars

    @property
    def model(self):
        """The underlying (lazily loaded, cached) SparseTextEmbedding."""
        return _load_model(self.model_id)

    def encode_passages(self, texts: list[str]) -> list[models.SparseVector]:
        """Encode chunk texts (document side) -> Qdrant SparseVectors."""
        prepared = [prepare_text(t, self.max_chars) for t in texts]
        return [_to_sparse_vector(e) for e in self.model.embed(prepared)]

    def encode_query(self, query: str) -> models.SparseVector:
        """Encode one query (query side: term presence, no BM25 tf saturation)."""
        prepared = prepare_text(query, self.max_chars)
        embedding = next(iter(self.model.query_embed(prepared)))
        return _to_sparse_vector(embedding)
