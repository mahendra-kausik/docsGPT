"""Dense-only baseline evaluation over the gold set (Layer 3).

This produces the reference numbers every later layer is measured against (PLAN
§4 Layer 3 gate): retrieval metrics on the verified gold set, saved to a results
file that records the exact config + git SHA + judge (none, for retrieval) so the
run reproduces (CLAUDE.md §6). Hybrid + rerank (Layer 4) re-runs this same harness
to build the before/after ablation table.

Run:  ./tasks.ps1 eval        (needs the live Qdrant cluster)
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src.config import PROJECT_ROOT, get_settings
from src.eval import metrics as M
from src.eval.gold import load_gold
from src.retrieval.search import DenseRetriever, HybridRetriever, RerankRetriever

logger = logging.getLogger(__name__)

# Pipeline id -> (retriever class, results-file label recorded in the run metadata).
_PIPELINES = {
    "dense": (DenseRetriever, "dense-only"),
    "hybrid": (HybridRetriever, "hybrid-rrf"),
    "rerank": (RerankRetriever, "hybrid-rerank"),
}


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _git_sha() -> str:
    """Short git SHA of the working tree, for reproducibility (best-effort)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 — reproducibility metadata, never fatal
        return "unknown"


def run_eval(
    pipeline: str = "dense", gold_path: str | None = None, top_k: int | None = None
) -> dict:
    """Score a retrieval pipeline over the gold set and return a results dict.

    pipeline: "dense" (Layer 2 baseline) or "hybrid" (Layer 4a dense+BM25 RRF). Both
    run the identical gold set + metrics so the results files form a clean ablation.
    """
    if pipeline not in _PIPELINES:
        raise SystemExit(f"Unknown pipeline {pipeline!r}. Choose: {', '.join(_PIPELINES)}")
    retriever_cls, pipeline_label = _PIPELINES[pipeline]

    s = get_settings()
    gold_file = gold_path or s.gold_jsonl
    items = load_gold(gold_file)
    scored = [it for it in items if it.is_scored]
    n_unanswerable = sum(1 for it in items if it.status == "unanswerable")

    if not scored:
        raise SystemExit(
            f"No scored gold items in {gold_file}. Build the gold set first: "
            "./tasks.ps1 propose -> edit review.md -> ./tasks.ps1 compile-gold"
        )

    # Retrieve deep enough to cover the largest reported cutoff.
    k = top_k or max((*M.DEFAULT_KS, M.MRR_K))
    retriever = retriever_cls()

    per_query: list[dict[str, float]] = []
    per_query_detail: list[dict] = []
    latencies: list[float] = []
    for it in scored:
        hits, ms = retriever.search(it.question, top_k=k)
        latencies.append(ms)
        ranked_ids = [h.id for h in hits]
        q_metrics = M.score_query(ranked_ids, it.gold_chunk_ids)
        per_query.append(q_metrics)
        per_query_detail.append(
            {
                "qid": it.qid,
                "gold_chunk_ids": it.gold_chunk_ids,
                "top_ids": ranked_ids,
                "metrics": q_metrics,
            }
        )

    agg = M.aggregate(per_query)

    # Per-source breakdown: real (forum) is the honest headline, synthetic is
    # augmentation and tends to score higher (questions share vocab with their gold
    # chunk) — reporting them apart makes that gap visible rather than hidden (D-025).
    by_source: dict[str, dict[str, float]] = {}
    sources = sorted({it.source for it in scored})
    if len(sources) > 1:
        for src in sources:
            idx = [i for i, it in enumerate(scored) if it.source == src]
            by_source[src] = {
                "n": len(idx),
                **M.aggregate([per_query[i] for i in idx]),
            }

    latencies_sorted = sorted(latencies)

    def _pct(p: float) -> float:
        idx = min(len(latencies_sorted) - 1, int(round(p * (len(latencies_sorted) - 1))))
        return round(latencies_sorted[idx], 1)

    uses_sparse = pipeline in ("hybrid", "rerank")
    config = {
        "embedding_model": s.embedding_model,
        "query_instruction": s.query_instruction,
        "qdrant_collection": s.qdrant_hybrid_collection if uses_sparse else s.qdrant_collection,
        "vector_distance": s.vector_distance,
        "retrieve_top_k": k,
        "ks": list(M.DEFAULT_KS),
        "mrr_k": M.MRR_K,
    }
    if uses_sparse:  # record the sparse half + the RRF constant that produced these numbers
        config["sparse_model"] = s.sparse_model
        config["rrf_k"] = s.rrf_k
    if pipeline == "rerank":  # record the reranker + the fused pool depth it rescored
        config["reranker_model"] = s.reranker_model
        config["rerank_pool"] = s.retrieve_top_k

    return {
        "run": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_sha": _git_sha(),
            "pipeline": pipeline_label,  # dense-only (Layer 3) or hybrid-rrf (Layer 4a)
            "judge_model": None,  # retrieval metrics need no LLM judge (PLAN §6)
        },
        "config": config,
        "gold": {
            "file": gold_file,
            "scored_items": len(scored),
            "unanswerable_dropped": n_unanswerable,
            "total_items": len(items),
        },
        "metrics": {name: round(val, 4) for name, val in agg.items()},
        "metrics_by_source": {
            src: {k: (v if k == "n" else round(v, 4)) for k, v in d.items()}
            for src, d in by_source.items()
        },
        "latency_ms": {
            "p50": _pct(0.50),
            "p95": _pct(0.95),
            "mean": round(statistics.fmean(latencies), 1),
        },
        "per_query": per_query_detail,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval eval over the gold set (dense or hybrid).")
    ap.add_argument(
        "--pipeline", choices=list(_PIPELINES), default="dense",
        help="retrieval pipeline: dense (Layer 2 baseline) or hybrid (Layer 4a)",
    )
    ap.add_argument("--gold", default=None, help="gold.jsonl path (default from config)")
    ap.add_argument(
        "--top-k", type=int, default=None, help="retrieval depth (default = max cutoff)"
    )
    ap.add_argument("--no-save", action="store_true", help="print only, don't write a results file")
    args = ap.parse_args()
    s = get_settings()

    results = run_eval(args.pipeline, args.gold, args.top_k)

    # Headline summary to stdout.
    m = results["metrics"]
    g = results["gold"]
    print(
        f"\n{results['run']['pipeline']}  ({g['scored_items']} scored / "
        f"{g['unanswerable_dropped']} unanswerable / {g['total_items']} total)"
    )
    headline = ("recall@1", "recall@5", "recall@10", "mrr@3", "ndcg@10", "hit@5")
    for name in headline:
        if name in m:
            print(f"  {name:<10} {m[name]:.4f}")
    print(f"  latency p50/p95 {results['latency_ms']['p50']}/{results['latency_ms']['p95']} ms")

    for src, d in results["metrics_by_source"].items():
        cols = "  ".join(f"{name}={d[name]:.3f}" for name in ("recall@5", "mrr@3") if name in d)
        print(f"  [{src:<9} n={int(d['n']):>3}]  {cols}")

    if not args.no_save:
        results_dir = _abs(s.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = results_dir / f"eval_{args.pipeline}_{stamp}.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
