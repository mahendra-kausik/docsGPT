"""RRF k-sweep over the hybrid retriever (Layer 4a ablation, D-027/D-028).

Why: the hybrid gate (k=60) improved overall but regressed the real forum slice
(D-028). k is the RRF rank constant — small k trusts each retriever's top hit,
large k trusts consensus. This sweeps k to answer: is the real-slice regression
k-driven (a better k recovers it) or fundamental to BM25 on NL questions (every k
regresses -> the Layer 4b reranker is the real fix)?

Retrieval is done ONCE per query at the gate's depth (max reported cutoff), then
fused at every k, so the k=60 row reproduces the committed hybrid ablation exactly
and the only thing varying is the merge. Run:  ./tasks.ps1 sweep-rrf
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from src.config import PROJECT_ROOT, get_settings
from src.eval import metrics as M
from src.eval.gold import load_gold
from src.retrieval.embedder import Embedder
from src.retrieval.fusion import rrf_fuse
from src.retrieval.index import get_client
from src.retrieval.search import _to_hit
from src.retrieval.sparse import SparseEmbedder

DEFAULT_KS = (2, 10, 30, 60, 100, 200)


def _retrieve_lists(items, depth: int) -> list[tuple[list[str], list[str]]]:
    """For each gold item, fetch (dense_ids, sparse_ids) once at the given depth."""
    s = get_settings()
    coll = s.qdrant_hybrid_collection
    embedder, sparse, client = Embedder(), SparseEmbedder(), get_client()
    out: list[tuple[list[str], list[str]]] = []
    for it in items:
        qvec = embedder.encode_query(it.question)
        qsparse = sparse.encode_query(it.question)
        dense = client.query_points(
            collection_name=coll, query=qvec, using="dense", limit=depth, with_payload=True
        )
        spr = client.query_points(
            collection_name=coll, query=qsparse, using="sparse", limit=depth, with_payload=True
        )
        out.append(
            ([_to_hit(p).id for p in dense.points], [_to_hit(p).id for p in spr.points])
        )
    return out


def _score_at_k(items, lists, k: int) -> dict:
    """Fuse every query's two lists at this k and aggregate metrics overall + per source."""
    per_query = []
    for (dense_ids, sparse_ids), it in zip(lists, items, strict=True):
        fused = rrf_fuse([dense_ids, sparse_ids], k=k)
        ranked = [cid for cid, _ in fused]
        per_query.append(M.score_query(ranked, it.gold_chunk_ids))
    agg = M.aggregate(per_query)
    by_source = {}
    for src in sorted({it.source for it in items}):
        idx = [i for i, it in enumerate(items) if it.source == src]
        by_source[src] = {"n": len(idx), **M.aggregate([per_query[i] for i in idx])}
    return {"overall": agg, "by_source": by_source}


def run_sweep(ks=DEFAULT_KS, gold_path: str | None = None, depth: int | None = None) -> dict:
    s = get_settings()
    items = [it for it in load_gold(gold_path or s.gold_jsonl) if it.is_scored]
    depth = depth or max((*M.DEFAULT_KS, M.MRR_K))  # gate depth, so k=60 reproduces
    lists = _retrieve_lists(items, depth)
    results = {k: _score_at_k(items, lists, k) for k in ks}
    return {
        "run": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "pipeline": "hybrid-rrf-ksweep",
            "retrieval_depth": depth,
            "scored_items": len(items),
        },
        "config": {
            "embedding_model": s.embedding_model,
            "sparse_model": s.sparse_model,
            "qdrant_collection": s.qdrant_hybrid_collection,
            "ks": list(ks),
        },
        "sweep": {str(k): v for k, v in results.items()},
    }


def _print_table(res: dict) -> None:
    sweep = res["sweep"]
    ks = [int(k) for k in sweep]
    print(f"\nRRF k-sweep  (n={res['run']['scored_items']}, depth={res['run']['retrieval_depth']})")
    print("  metric / slice          " + "".join(f"{('k='+str(k)):>9}" for k in ks))
    rows = [
        ("recall@5  overall", "overall", "recall@5"),
        ("mrr@3     overall", "overall", "mrr@3"),
        ("recall@5  forum", "forum", "recall@5"),
        ("mrr@3     forum", "forum", "mrr@3"),
        ("recall@10 forum", "forum", "recall@10"),
        ("recall@5  synthetic", "synthetic", "recall@5"),
        ("mrr@3     synthetic", "synthetic", "mrr@3"),
    ]
    for label, slice_, metric in rows:
        cells = []
        for k in ks:
            block = sweep[str(k)]
            d = block["overall"] if slice_ == "overall" else block["by_source"].get(slice_, {})
            cells.append(f"{d.get(metric, float('nan')):>9.3f}")
        print(f"  {label:<22}" + "".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description="RRF k-sweep over the hybrid retriever.")
    ap.add_argument("--ks", default=None, help="comma-separated k values (default: see DEFAULT_KS)")
    ap.add_argument("--gold", default=None, help="gold.jsonl path (default from config)")
    ap.add_argument("--no-save", action="store_true", help="print only, don't save")
    args = ap.parse_args()
    s = get_settings()
    ks = tuple(int(x) for x in args.ks.split(",")) if args.ks else DEFAULT_KS

    res = run_sweep(ks, args.gold)
    _print_table(res)

    if not args.no_save:
        out_dir = PROJECT_ROOT / s.results_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"sweep_rrf_{stamp}.json"
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
