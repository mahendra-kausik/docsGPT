"""Build the hybrid (dense + BM25 sparse) index in Qdrant Cloud (Layer 4a, D-026).

Why a *separate* docs_hybrid collection: it leaves docs_dense intact as the
reproducible Layer 2 baseline (clean rollback; both stay queryable). Why *scroll*
the dense vectors out of docs_dense instead of re-embedding: the ~30-min bge encode
is reused verbatim — only the BM25 sparse vector is newly computed — which keeps the
re-index cheap and free-tier-friendly (D-026). The sparse vector carries BM25's
term-frequency component; IDF is applied server-side via Modifier.IDF.

Run:  ./tasks.ps1 index-hybrid                 # reuse dense vectors from docs_dense
      ./tasks.ps1 index-hybrid --limit 200 --no-recreate
"""

from __future__ import annotations

import argparse
import time

from qdrant_client import models

from src.config import get_settings
from src.retrieval.index import get_client
from src.retrieval.sparse import SparseEmbedder

_DENSE = "dense"
_SPARSE = "sparse"


def _ensure_hybrid_collection(
    client, name: str, dim: int, distance: str, recreate: bool
) -> None:
    """Create docs_hybrid with named `dense` + `sparse` vectors (dropping any existing)."""
    dist = models.Distance[distance.upper()]
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={_DENSE: models.VectorParams(size=dim, distance=dist)},
            # Modifier.IDF: Qdrant computes BM25's IDF from corpus stats at query time,
            # so stored/query sparse vectors carry only the term-frequency component.
            sparse_vectors_config={
                _SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )


def build_hybrid_index(*, recreate: bool = True, limit: int | None = None) -> dict:
    """Copy dense vectors from docs_dense into docs_hybrid, adding a BM25 sparse vector."""
    s = get_settings()
    client = get_client()
    sparse = SparseEmbedder()

    src_name = s.qdrant_collection
    dst_name = s.qdrant_hybrid_collection
    if not client.collection_exists(src_name):
        raise SystemExit(
            f"Source collection {src_name!r} not found. Build the dense index first: "
            "./tasks.ps1 index"
        )

    dim = client.get_collection(src_name).config.params.vectors.size
    _ensure_hybrid_collection(client, dst_name, dim, s.vector_distance, recreate)

    t0 = time.perf_counter()
    n = 0
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=src_name,
            limit=s.upsert_batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,  # reuse the existing dense vectors (no re-embed)
        )
        if not records:
            break

        texts = [(r.payload or {}).get("text", "") for r in records]
        sparse_vecs = sparse.encode_passages(texts)
        points = [
            models.PointStruct(
                id=r.id,
                vector={_DENSE: r.vector, _SPARSE: sv},
                payload=r.payload,
            )
            for r, sv in zip(records, sparse_vecs, strict=True)
        ]
        client.upsert(collection_name=dst_name, points=points, wait=True)
        n += len(points)
        print(f"  upserted {n}", flush=True)

        if offset is None or (limit is not None and n >= limit):
            break

    elapsed = time.perf_counter() - t0
    count = client.count(dst_name, exact=True).count
    return {
        "collection": dst_name,
        "source": src_name,
        "sparse_model": sparse.model_id,
        "dim": dim,
        "points_upserted": n,
        "collection_count": count,
        "seconds": round(elapsed, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the hybrid index in Qdrant Cloud.")
    ap.add_argument("--limit", type=int, default=None, help="index only the first N points")
    ap.add_argument("--no-recreate", action="store_true", help="keep the existing collection")
    args = ap.parse_args()

    stats = build_hybrid_index(recreate=not args.no_recreate, limit=args.limit)
    print("\nHybrid index build complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
