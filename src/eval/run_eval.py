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
from src.retrieval.search import DenseRetriever

logger = logging.getLogger(__name__)


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


def run_dense_eval(gold_path: str | None = None, top_k: int | None = None) -> dict:
    """Score the dense retriever over the gold set and return a results dict."""
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
    retriever = DenseRetriever()

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
    latencies_sorted = sorted(latencies)

    def _pct(p: float) -> float:
        idx = min(len(latencies_sorted) - 1, int(round(p * (len(latencies_sorted) - 1))))
        return round(latencies_sorted[idx], 1)

    return {
        "run": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_sha": _git_sha(),
            "pipeline": "dense-only",  # Layer 3 baseline; Layer 4 adds hybrid/rerank
            "judge_model": None,  # retrieval metrics need no LLM judge (PLAN §6)
        },
        "config": {
            "embedding_model": s.embedding_model,
            "query_instruction": s.query_instruction,
            "qdrant_collection": s.qdrant_collection,
            "vector_distance": s.vector_distance,
            "retrieve_top_k": k,
            "ks": list(M.DEFAULT_KS),
            "mrr_k": M.MRR_K,
        },
        "gold": {
            "file": gold_file,
            "scored_items": len(scored),
            "unanswerable_dropped": n_unanswerable,
            "total_items": len(items),
        },
        "metrics": {name: round(val, 4) for name, val in agg.items()},
        "latency_ms": {
            "p50": _pct(0.50),
            "p95": _pct(0.95),
            "mean": round(statistics.fmean(latencies), 1),
        },
        "per_query": per_query_detail,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Dense-only baseline eval over the gold set.")
    ap.add_argument("--gold", default=None, help="gold.jsonl path (default from config)")
    ap.add_argument(
        "--top-k", type=int, default=None, help="retrieval depth (default = max cutoff)"
    )
    ap.add_argument("--no-save", action="store_true", help="print only, don't write a results file")
    args = ap.parse_args()
    s = get_settings()

    results = run_dense_eval(args.gold, args.top_k)

    # Headline summary to stdout.
    m = results["metrics"]
    g = results["gold"]
    print(
        f"\nDense-only baseline  ({g['scored_items']} scored / "
        f"{g['unanswerable_dropped']} unanswerable / {g['total_items']} total)"
    )
    for name in ("recall@1", "recall@5", "recall@10", "mrr@3", "ndcg@10", "hit@5"):
        if name in m:
            print(f"  {name:<10} {m[name]:.4f}")
    print(f"  latency p50/p95 {results['latency_ms']['p50']}/{results['latency_ms']['p95']} ms")

    if not args.no_save:
        results_dir = _abs(s.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = results_dir / f"eval_dense_{stamp}.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
