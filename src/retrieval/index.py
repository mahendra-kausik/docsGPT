"""Build the dense vector index in the deployed Qdrant cluster (Layer 2).

Why: the chunked corpus (chunks.jsonl — the durable source of truth, D-004) is
embedded once with bge-small and pushed to Qdrant Cloud so retrieval runs against
a *deployed* store (the Layer 2 gate), not a local index. Re-runnable: it recreates
the collection so a re-index reproduces the same state (the D-004 re-index story).

Run:  ./tasks.ps1 index            # full corpus
      ./tasks.ps1 index --limit 200 --no-recreate
"""

from __future__ import annotations

import argparse
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from qdrant_client import QdrantClient, models

from src.config import get_settings
from src.ingest.models import Chunk
from src.retrieval.embedder import Embedder

# Stable namespace so a chunk's string id maps to the SAME Qdrant point id on every
# re-index (Qdrant point ids must be uint64 or UUID; our ids are short hex hashes).
_ID_NAMESPACE = uuid.UUID("6f9d1b2e-3c4a-5d6e-7f80-91a2b3c4d5e6")


def point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk id, so upserts are idempotent across runs."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def get_client() -> QdrantClient:
    """Client for the deployed Qdrant Cloud cluster (url + key from .env)."""
    s = get_settings()
    if not s.qdrant_url or not s.qdrant_api_key:
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY not set in .env")
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=60)


def _iter_chunks(path: Path, limit: int | None = None) -> Iterator[Chunk]:
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield Chunk.model_validate_json(line)


def _ensure_collection(
    client: QdrantClient, name: str, dim: int, distance: str, recreate: bool
) -> None:
    """Create the collection (dropping any existing one when recreate=True)."""
    dist = models.Distance[distance.upper()]
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=dist),
        )


def build_index(*, recreate: bool = True, limit: int | None = None) -> dict:
    """Embed all chunks and upsert them into the deployed Qdrant collection."""
    s = get_settings()
    embedder = Embedder()
    client = get_client()
    dim = embedder.dim  # forces model load; derived so index/query dims agree (D-021)

    _ensure_collection(client, s.qdrant_collection, dim, s.vector_distance, recreate)

    chunks = list(_iter_chunks(Path(s.corpus_jsonl), limit))
    t0 = time.perf_counter()
    n = 0
    for start in range(0, len(chunks), s.upsert_batch_size):
        batch = chunks[start : start + s.upsert_batch_size]
        vectors = embedder.encode_passages([c.text for c in batch], show_progress=False)
        points = [
            models.PointStruct(id=point_id(c.id), vector=v.tolist(), payload=c.model_dump())
            for c, v in zip(batch, vectors, strict=True)
        ]
        client.upsert(collection_name=s.qdrant_collection, points=points, wait=True)
        n += len(points)
        print(f"  upserted {n}/{len(chunks)}", flush=True)

    elapsed = time.perf_counter() - t0
    count = client.count(s.qdrant_collection, exact=True).count
    return {
        "collection": s.qdrant_collection,
        "model": embedder.model_id,
        "dim": dim,
        "distance": s.vector_distance,
        "chunks_indexed": n,
        "collection_count": count,
        "seconds": round(elapsed, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the dense index in Qdrant Cloud.")
    ap.add_argument("--limit", type=int, default=None, help="index only the first N chunks")
    ap.add_argument("--no-recreate", action="store_true", help="keep the existing collection")
    args = ap.parse_args()

    stats = build_index(recreate=not args.no_recreate, limit=args.limit)
    print("\nIndex build complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
