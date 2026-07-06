"""Dense top-k retrieval against the deployed Qdrant cluster (Layer 2).

Why separate from indexing: this is the query path the eval harness (Layer 3) and
the agent (Layer 5) call. It returns scored hits with full provenance payload so a
caller can rerank (Layer 4) and cite (Layer 5). Latency is logged/returned per the
Layer 2 gate.

Run:  ./tasks.ps1 search "how do I stream tokens from a chat model?"
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field

from src.config import get_settings
from src.retrieval.embedder import Embedder
from src.retrieval.index import get_client

logger = logging.getLogger(__name__)


@dataclass
class Hit:
    """One retrieved chunk with its similarity score and provenance."""

    id: str
    score: float
    text: str
    source_url: str
    heading_path: str
    payload: dict = field(repr=False, default_factory=dict)


def _to_hit(scored) -> Hit:
    p = scored.payload or {}
    return Hit(
        id=p.get("id", str(scored.id)),
        score=float(scored.score),
        text=p.get("text", ""),
        source_url=p.get("source_url", ""),
        heading_path=p.get("heading_path", ""),
        payload=p,
    )


class DenseRetriever:
    """bge-small query → Qdrant cosine search → scored, cited hits."""

    def __init__(self) -> None:
        self.s = get_settings()
        self.embedder = Embedder()
        self.client = get_client()

    def search(self, query: str, top_k: int | None = None) -> tuple[list[Hit], float]:
        """Return (hits, latency_ms) for a dense top-k search over the corpus."""
        top_k = top_k or self.s.retrieve_top_k
        t0 = time.perf_counter()
        qvec = self.embedder.encode_query(query)
        response = self.client.query_points(
            collection_name=self.s.qdrant_collection,
            query=qvec,
            limit=top_k,
            with_payload=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        hits = [_to_hit(p) for p in response.points]
        logger.info(
            "dense_search q=%r k=%d -> %d hits in %.1f ms", query[:60], top_k, len(hits), elapsed_ms
        )
        return hits, elapsed_ms


def main() -> None:
    ap = argparse.ArgumentParser(description="Dense search over the deployed Qdrant corpus.")
    ap.add_argument("query", nargs="+", help="the search query")
    ap.add_argument("-k", "--top-k", type=int, default=5, help="how many hits to show")
    args = ap.parse_args()

    query = " ".join(args.query)
    hits, elapsed_ms = DenseRetriever().search(query, top_k=args.top_k)
    print(f'\nQuery: {query!r}')
    print(f"Returned {len(hits)} hits in {elapsed_ms:.1f} ms\n")
    for i, h in enumerate(hits, 1):
        snippet = " ".join(h.text.split())[:160]
        print(f"{i}. score={h.score:.4f}  {h.heading_path}")
        print(f"   {h.source_url}")
        print(f"   {snippet}\n")


if __name__ == "__main__":
    main()
