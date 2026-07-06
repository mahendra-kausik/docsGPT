"""Dense embedding for retrieval (Layer 2).

Why a thin wrapper: index-time and query-time vectors must live in one space, so
the whole codebase embeds through a single model + normalization + input-hygiene
path (a mismatch here is a classic RAG bug — D-005). The model id and the bge
query instruction come from config, so the bge-small/bge-base ablation (D-005) is
a one-line change.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from src.config import get_settings

# Long base64 blobs (data-URI images / inlined binaries) sometimes survive MDX
# cleaning inside doc code examples (D-017 tail). They carry no semantic signal
# and would otherwise consume the whole 512-token budget, so we strip them before
# encoding (D-022).
_DATA_URI = re.compile(r"data:[\w.+-]+/[\w.+-]+;base64,[A-Za-z0-9+/=]+")
_B64_RUN = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")
_BINARY_PLACEHOLDER = "[binary omitted]"


def scrub_binary(text: str) -> str:
    """Replace data-URI and long base64 runs with a placeholder (D-022)."""
    text = _DATA_URI.sub(_BINARY_PLACEHOLDER, text)
    text = _B64_RUN.sub(_BINARY_PLACEHOLDER, text)
    return text


def prepare_text(text: str, max_chars: int) -> str:
    """Scrub binary noise, then hard-cap length before the tokenizer sees it."""
    return scrub_binary(text)[:max_chars]


@lru_cache(maxsize=2)
def _load_model(model_id: str):
    """Load a SentenceTransformer once per id (cached). Imported lazily so the
    scrub helpers stay importable without torch installed."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


class Embedder:
    """Encodes text into L2-normalized dense vectors for cosine retrieval."""

    def __init__(self, model_id: str | None = None, max_chars: int | None = None) -> None:
        s = get_settings()
        self.model_id = model_id or s.embedding_model
        self.max_chars = max_chars or s.embed_max_chars
        self.batch_size = s.embed_batch_size
        self.query_instruction = s.query_instruction

    @property
    def model(self):
        """The underlying (lazily loaded, cached) SentenceTransformer."""
        return _load_model(self.model_id)

    @property
    def dim(self) -> int:
        """Embedding dimension read from the model so index and query always agree."""
        model = self.model
        # sentence-transformers renamed this in 5.x; support both without warnings.
        if hasattr(model, "get_embedding_dimension"):
            return int(model.get_embedding_dimension())
        return int(model.get_sentence_embedding_dimension())

    def encode_passages(
        self, texts: list[str], *, batch_size: int | None = None, show_progress: bool = False
    ) -> np.ndarray:
        """Encode chunk texts (no instruction prefix) → (n, dim) normalized array."""
        prepared = [prepare_text(t, self.max_chars) for t in texts]
        return self.model.encode(
            prepared,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def encode_query(self, query: str) -> list[float]:
        """Encode one query (bge instruction prepended) → python list for Qdrant."""
        text = f"{self.query_instruction}{query}" if self.query_instruction else query
        vec = self.model.encode(
            [prepare_text(text, self.max_chars)],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return vec.tolist()
