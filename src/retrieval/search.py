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
from dataclasses import dataclass, field, replace

from src.config import get_settings
from src.retrieval.embedder import Embedder
from src.retrieval.fusion import rrf_fuse
from src.retrieval.index import get_client
from src.retrieval.rerank import Reranker, apply_rerank
from src.retrieval.sparse import SparseEmbedder

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


class HybridRetriever:
    """Dense + BM25 sparse retrieval fused with client-side RRF (Layer 4a, D-026/D-027).

    Two named-vector queries against docs_hybrid (dense cosine + sparse BM25) are fused
    with rrf_fuse (k from config, tunable) rather than Qdrant's server-side FusionQuery,
    so k is honoured and sweepable. Returns the same Hit shape as DenseRetriever so the
    eval harness and agent are retriever-agnostic; hit.score becomes the RRF score.
    """

    def __init__(self) -> None:
        self.s = get_settings()
        self.embedder = Embedder()
        self.sparse = SparseEmbedder()
        self.client = get_client()

    def search(self, query: str, top_k: int | None = None) -> tuple[list[Hit], float]:
        """Return (hits, latency_ms) for a hybrid RRF-fused top-k search."""
        top_k = top_k or self.s.retrieve_top_k
        coll = self.s.qdrant_hybrid_collection
        t0 = time.perf_counter()
        qvec = self.embedder.encode_query(query)
        qsparse = self.sparse.encode_query(query)
        dense_resp = self.client.query_points(
            collection_name=coll, query=qvec, using="dense", limit=top_k, with_payload=True
        )
        sparse_resp = self.client.query_points(
            collection_name=coll, query=qsparse, using="sparse", limit=top_k, with_payload=True
        )
        # Fuse on chunk id (payload id) — the space the gold set is scored in.
        dense_hits = [_to_hit(p) for p in dense_resp.points]
        sparse_hits = [_to_hit(p) for p in sparse_resp.points]
        by_id = {h.id: h for h in (*dense_hits, *sparse_hits)}
        fused = rrf_fuse(
            [[h.id for h in dense_hits], [h.id for h in sparse_hits]], k=self.s.rrf_k
        )
        hits = [replace(by_id[cid], score=score) for cid, score in fused[:top_k]]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "hybrid_search q=%r k=%d dense=%d sparse=%d -> %d fused in %.1f ms",
            query[:60], top_k, len(dense_hits), len(sparse_hits), len(hits), elapsed_ms,
        )
        return hits, elapsed_ms


class RerankRetriever:
    """Hybrid retrieval + cross-encoder reranking (Layer 4b, D-007/D-029).

    Retrieves a DEEP fused pool (retrieve_top_k candidates) so correct-but-demoted
    chunks are present (the fix for the D-028 regression), then a cross-encoder
    rescores the pool and the top_k are returned. Same Hit shape as the other
    retrievers, so the eval harness is retriever-agnostic; hit.score becomes the
    cross-encoder score.
    """

    def __init__(self) -> None:
        self.s = get_settings()
        self.hybrid = HybridRetriever()
        self.reranker = Reranker()

    def search(self, query: str, top_k: int | None = None) -> tuple[list[Hit], float]:
        """Return (hits, latency_ms) for hybrid-retrieve → cross-encoder rerank."""
        top_k = top_k or self.s.rerank_top_n
        t0 = time.perf_counter()
        # Deep candidate pool: the cross-encoder can only rescue a chunk that survived
        # fusion into the pool, so pool depth (retrieve_top_k) >> the returned top_k.
        pool, _ = self.hybrid.search(query, top_k=self.s.retrieve_top_k)
        scores = self.reranker.score(query, [h.text for h in pool])
        hits = apply_rerank(pool, scores, top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "rerank_search q=%r pool=%d -> top %d in %.1f ms",
            query[:60], len(pool), len(hits), elapsed_ms,
        )
        return hits, elapsed_ms


class DecomposedRetriever:
    """Query decomposition + multi-query hybrid retrieval fused with RRF (Layer 5b-i).

    Splits the question into 1-3 sub-queries (Groq, D-037), retrieves each over the
    hybrid index, and fuses the per-sub-query result lists with RRF — the intended fix
    for the real-slice weakness (D-031), where a single query misses one facet of a
    multi-part question. A single sub-query degrades exactly to HybridRetriever. Same
    Hit shape as the others so the eval harness stays retriever-agnostic.
    """

    def __init__(self, gateway=None) -> None:
        self.s = get_settings()
        self.hybrid = HybridRetriever()
        self.gateway = gateway  # decomposition LLM; None -> Groq default

    def search(self, query: str, top_k: int | None = None) -> tuple[list[Hit], float]:
        """Return (hits, latency_ms) for decomposed multi-query RRF retrieval."""
        from src.agent.decompose import decompose

        top_k = top_k or self.s.retrieve_top_k
        t0 = time.perf_counter()
        subqueries = decompose(query, self.gateway)
        ranked_lists: list[list[str]] = []
        by_id: dict[str, Hit] = {}
        for sq in subqueries:
            hits, _ = self.hybrid.search(sq, top_k=top_k)
            for h in hits:
                by_id[h.id] = h
            ranked_lists.append([h.id for h in hits])
        fused = rrf_fuse(ranked_lists, k=self.s.rrf_k)
        hits = [replace(by_id[cid], score=score) for cid, score in fused[:top_k]]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "decomposed_search q=%r -> %d subqueries -> %d fused in %.1f ms",
            query[:60], len(subqueries), len(hits), elapsed_ms,
        )
        return hits, elapsed_ms


def main() -> None:
    ap = argparse.ArgumentParser(description="Search over the deployed Qdrant corpus.")
    ap.add_argument("query", nargs="+", help="the search query")
    ap.add_argument("-k", "--top-k", type=int, default=5, help="how many hits to show")
    ap.add_argument("--hybrid", action="store_true", help="use hybrid (dense+BM25 RRF) retrieval")
    ap.add_argument("--rerank", action="store_true", help="hybrid + cross-encoder rerank")
    ap.add_argument("--decomposed", action="store_true", help="query decomposition + multi-query")
    args = ap.parse_args()

    query = " ".join(args.query)
    if args.decomposed:
        retriever = DecomposedRetriever()
    elif args.rerank:
        retriever = RerankRetriever()
    elif args.hybrid:
        retriever = HybridRetriever()
    else:
        retriever = DenseRetriever()
    hits, elapsed_ms = retriever.search(query, top_k=args.top_k)
    print(f'\nQuery: {query!r}')
    print(f"Returned {len(hits)} hits in {elapsed_ms:.1f} ms\n")
    for i, h in enumerate(hits, 1):
        snippet = " ".join(h.text.split())[:160]
        print(f"{i}. score={h.score:.4f}  {h.heading_path}")
        print(f"   {h.source_url}")
        print(f"   {snippet}\n")


if __name__ == "__main__":
    main()
