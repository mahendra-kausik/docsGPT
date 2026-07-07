"""Reranker bake-off over the hybrid candidate pool (Layer 4c, D-032).

Why: Layer 4b rejected reranking, but on incomplete evidence — MiniLM-L6 (PyTorch)
was measured and failed on quality, while bge-reranker-base was killed on *latency*
(PyTorch path, ~25 s/query) with its quality never measured (D-030). fastembed exposes
an ONNX reranker (TextCrossEncoder) ~2-4x faster on CPU, so we can now measure the
*quality* of several stronger/domain-appropriate rerankers and read off a quality-vs-
latency Pareto — focused on the REAL forum slice, where reranking has hurt so far.

Method (mirrors sweep_rrf): retrieve the deep fused pool ONCE per query, then rerank
that same pool with each model, so retrieval cost is paid once and the reported latency
is rerank-only. Latency is LOCAL CPU fp32 ONNX — a floor a quantized/served path beats.

Run:  ./tasks.ps1 bakeoff            # all candidates
      ./tasks.ps1 bakeoff --models Xenova/ms-marco-MiniLM-L-12-v2
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime

from src.config import PROJECT_ROOT, get_settings
from src.eval import metrics as M
from src.eval.gold import load_gold
from src.retrieval.embedder import prepare_text
from src.retrieval.rerank import apply_rerank
from src.retrieval.search import HybridRetriever

# Ordered light -> heavy so the fast, deployable candidates report first and the big
# (~1 GB) models run last; each result is saved incrementally.
CANDIDATES = [
    "Xenova/ms-marco-MiniLM-L-6-v2",          # ONNX twin of the L4b PyTorch run (control)
    "Xenova/ms-marco-MiniLM-L-12-v2",         # bigger MiniLM, still deployable-fast
    "BAAI/bge-reranker-base",                 # strong; quality never measured (D-030 gap)
    "jinaai/jina-reranker-v2-base-multilingual",  # strong + best domain fit for code docs
]


def _retrieve_pools(items, depth: int):
    """Fetch the deep fused hybrid pool once per query (id + text needed to rerank)."""
    hybrid = HybridRetriever()
    pools = []
    for it in items:
        hits, _ = hybrid.search(it.question, top_k=depth)
        pools.append(hits)
    return pools


def _aggregate_slices(items, per_query):
    """Overall + per-source aggregate for one model's per-query metrics."""
    out = {"overall": M.aggregate(per_query)}
    for src in sorted({it.source for it in items}):
        idx = [i for i, it in enumerate(items) if it.source == src]
        out[src] = {"n": len(idx), **M.aggregate([per_query[i] for i in idx])}
    return out


def _score_hybrid_baseline(items, pools, top_k: int):
    """No-rerank baseline: the fused RRF order itself (what we currently ship)."""
    per_query = [
        M.score_query([h.id for h in pool[:top_k]], it.gold_chunk_ids)
        for pool, it in zip(pools, items, strict=True)
    ]
    return {"latency_ms": {"p50": 0.0, "p95": 0.0}, **_aggregate_slices(items, per_query)}


def _score_model(model_name, items, pools, top_k, max_chars):
    """Rerank each pool with one model; return quality (overall+per source) + latency."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    enc = TextCrossEncoder(model_name=model_name)
    per_query, latencies = [], []
    for pool, it in zip(pools, items, strict=True):
        texts = [prepare_text(h.text, max_chars) for h in pool]
        t0 = time.perf_counter()
        scores = list(enc.rerank(it.question, texts))
        latencies.append((time.perf_counter() - t0) * 1000.0)
        hits = apply_rerank(pool, scores, top_k)
        per_query.append(M.score_query([h.id for h in hits], it.gold_chunk_ids))
    lat = sorted(latencies)

    def _p(p):
        return round(lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))], 1)

    return {
        "latency_ms": {"p50": _p(0.50), "p95": _p(0.95), "mean": round(statistics.fmean(lat), 1)},
        **_aggregate_slices(items, per_query),
    }


def run_bakeoff(
    models=CANDIDATES,
    gold_path: str | None = None,
    depth: int | None = None,
    synth_sample: int | None = None,
):
    s = get_settings()
    items = [it for it in load_gold(gold_path or s.gold_jsonl) if it.is_scored]
    if synth_sample is not None:
        # Keep every REAL forum item (the decision-critical slice) + a deterministic
        # synthetic sample, so the heavy 1 GB models finish in minutes on CPU (D-032).
        forum = [it for it in items if it.source != "synthetic"]
        synth = [it for it in items if it.source == "synthetic"][:synth_sample]
        items = forum + synth
    depth = depth or s.retrieve_top_k
    top_k = max((*M.DEFAULT_KS, M.MRR_K))
    pools = _retrieve_pools(items, depth)

    results = {
        "run": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "pipeline": "rerank-bakeoff",
            "pool_depth": depth,
            "rerank_max_chars": s.rerank_max_chars,
            "scored_items": len(items),
            "n_forum": sum(1 for it in items if it.source != "synthetic"),
            "note": "latency is local-CPU fp32 ONNX rerank-only (retrieval excluded)",
        },
        "models": {"hybrid-no-rerank": _score_hybrid_baseline(items, pools, top_k)},
    }
    for name in models:
        print(f"\n[bakeoff] reranking with {name} ...", flush=True)
        t0 = time.perf_counter()
        results["models"][name] = _score_model(name, items, pools, top_k, s.rerank_max_chars)
        print(f"[bakeoff] {name} done in {time.perf_counter() - t0:.0f}s", flush=True)
        _save(results, s)  # incremental: survive a long/interrupted run
    return results


def _save(results, s):
    out_dir = PROJECT_ROOT / s.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rerank_bakeoff_latest.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


def _print_table(results):
    models = results["models"]
    run = results["run"]
    print(f"\nReranker bake-off  (n={run['scored_items']}, pool={run['pool_depth']})")
    cols = f"{'r@5':>7}{'r@5-forum':>11}{'r@5-synth':>11}{'mrr@3':>8}{'ndcg@10':>9}{'p50ms':>9}"
    print(f"  {'model':<44}{cols}")
    for name, d in models.items():
        forum = d.get("forum", {}).get("recall@5", float("nan"))
        synth = d.get("synthetic", {}).get("recall@5", float("nan"))
        o = d["overall"]
        print(
            f"  {name:<44}{o['recall@5']:>7.3f}{forum:>11.3f}{synth:>11.3f}"
            f"{o['mrr@3']:>8.3f}{o['ndcg@10']:>9.3f}{d['latency_ms']['p50']:>9.1f}"
        )


def main():
    ap = argparse.ArgumentParser(description="ONNX reranker bake-off over the hybrid pool.")
    ap.add_argument("--models", nargs="*", default=None, help="model names (default: CANDIDATES)")
    ap.add_argument("--gold", default=None, help="gold.jsonl path (default from config)")
    ap.add_argument("--pool", type=int, default=None, help="rerank pool depth")
    ap.add_argument(
        "--synth-sample", type=int, default=None,
        help="keep all forum items + only this many synthetic (default: all)",
    )
    args = ap.parse_args()
    results = run_bakeoff(args.models or CANDIDATES, args.gold, args.pool, args.synth_sample)
    _print_table(results)
    print(f"\nSaved: {PROJECT_ROOT / get_settings().results_dir / 'rerank_bakeoff_latest.json'}")


if __name__ == "__main__":
    main()
