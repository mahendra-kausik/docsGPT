"""Layer 5d: RAGAS answer-quality eval over the AGENT's answers to the gold set (D-040).

Runs the full agent (retrieve -> synthesize -> verify -> cite, incl. 5c self-correction)
over a fixed, stratified sample of gold questions, then grades the answers with RAGAS:

  - Faithfulness + Answer Relevancy  -> every sampled item (the hallucination signal
    PLAN §6 wants read *together*: high faithfulness with low relevancy is a red flag).
  - Answer Correctness + Context Precision/Recall  -> only items that have a reference
    answer, i.e. the REAL forum slice (accepted forum answer = ground truth). Synthetic
    items are gold-by-construction chunks with no reference answer, so those metrics skip.

The judge is fixed to Groq-8B and routed through our gateway (backoff + cache) via the
adapters, so scores are comparable across runs (D-008/D-010) and re-scoring is free. The
generated dataset is written alongside the scores so a re-score needs no agent/LLM calls
(`--from-dataset`). Results record config + git SHA + judge model (CLAUDE.md §6).

Run:  ./tasks.ps1 ragas --sample 24
      ./tasks.ps1 ragas --from-dataset results/ragas_dataset_<stamp>.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src.config import PROJECT_ROOT, get_settings
from src.eval.gold import ForumSeed, GoldItem, accepted_answer_text, load_gold, load_seeds

# NOTE: importing ragas_adapters runs the vertexai compat shim before ragas is imported.
from src.eval.ragas_adapters import BGERagasEmbeddings, GatewayRagasLLM

FORUM_SOURCE = "forum"

# Fields carried per sample; the rest of a to_pandas() row is the RAGAS metric columns.
_INPUT_FIELDS = ("user_input", "response", "retrieved_contexts", "reference")


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _git_sha() -> str:
    """Short git SHA for reproducibility metadata (best-effort, never fatal)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def stratified_sample(items: list[GoldItem], n: int, seed: int) -> list[GoldItem]:
    """Pick n scored items, guaranteeing forum (real, reference-bearing) items are present.

    The reference-based metrics (Answer Correctness, Context Precision/Recall) only run on
    the forum slice, so we deliberately fill up to half the sample with forum items before
    topping up with synthetic — otherwise a random draw could starve those metrics.
    """
    scored = [it for it in items if it.is_scored]
    forum = [it for it in scored if it.source == FORUM_SOURCE]
    synthetic = [it for it in scored if it.source != FORUM_SOURCE]
    rng = random.Random(seed)
    rng.shuffle(forum)
    rng.shuffle(synthetic)
    n_forum = min(len(forum), max(1, n // 2))
    picked = forum[:n_forum] + synthetic[: n - n_forum]
    rng.shuffle(picked)
    return picked[:n]


def build_samples(
    items: list[GoldItem],
    seeds_by_id: dict[int, ForumSeed],
    *,
    synthesis_model: str | None = None,
) -> list[dict]:
    """Run the agent over each item and collect the (question, answer, contexts, reference).

    The agent is compiled ONCE and re-invoked so the retriever + gateways are shared across
    items (no re-instantiation per question). Reference answers come from the cached accepted
    forum answer for real items; synthetic items have no reference ("").

    ``synthesis_model`` overrides the agent's synthesis gateway for this run only (the graph
    default is Gemini, D-033). Used to route synthesis to Groq when Gemini's 20-req/day free
    cap is exhausted — recorded in the results so the graded pipeline is never misread.
    """
    from src.agent.graph import build_agent

    gateway = None
    if synthesis_model:
        from src.llm.gateway import LLMGateway

        gateway = LLMGateway(synthesis_model)
    agent = build_agent(gateway=gateway)
    settings = get_settings()
    # Stamp the ACTUAL synthesis model onto each sample so a later --from-dataset re-score
    # records the truth (the dataset alone can't otherwise say what produced the answers).
    synth_used = synthesis_model or settings.synthesis_model
    max_retries = settings.agent_max_retries
    samples = []
    for idx, it in enumerate(items, start=1):
        state = agent.invoke({"question": it.question, "retries": 0, "max_retries": max_retries})
        reference = ""
        if it.source == FORUM_SOURCE:
            seed = seeds_by_id.get(it.qid)
            if seed is not None:
                reference = accepted_answer_text(seed)
        samples.append(
            {
                "qid": it.qid,
                "source": it.source,
                "user_input": it.question,
                "response": state.get("answer", "").strip(),
                "retrieved_contexts": [h.text for h in state.get("chunks", [])],
                "reference": reference,
                "grounded": bool(state.get("grounded", True)),
                "retries": int(state.get("retries", 0)),
                "synthesis_model": synth_used,
            }
        )
        print(f"  [{idx}/{len(items)}] qid={it.qid} ({it.source}) retries={samples[-1]['retries']}")
    return samples


def _mean(values: list[float]) -> float | None:
    """Mean over non-null metric values (RAGAS emits NaN for a metric it couldn't score)."""
    nums = [v for v in values if isinstance(v, (int | float)) and v == v]  # v==v drops NaN
    return round(sum(nums) / len(nums), 4) if nums else None


def score_samples(samples: list[dict]) -> dict:
    """Grade the samples with RAGAS and return overall + per-source metric means."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        AnswerCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    from src.retrieval.embedder import prepare_text

    s = get_settings()
    llm = GatewayRagasLLM()
    embeddings = BGERagasEmbeddings()
    # Groq free tier is 6000 tokens/MINUTE and rejects any single request over it (413).
    # max_workers=1 serializes calls; max_retries + max_wait let RAGAS back off through the
    # per-minute 429s; a long timeout tolerates the resulting waits. The real fix for the
    # hard 413 is capping contexts below (fewer/shorter chunks per judge request).
    run_config = RunConfig(max_workers=1, timeout=600, max_retries=10, max_wait=120)

    def _cap_contexts(contexts: list[str]) -> list[str]:
        return [prepare_text(c, s.ragas_context_chars) for c in contexts[: s.ragas_context_k]]

    def _cap_field(key: str, value):
        # Contexts and the (often long) forum reference are truncated to keep each judge
        # request under Groq's 6000 TPM; the response/question are left intact.
        if key == "retrieved_contexts":
            return _cap_contexts(value)
        if key == "reference":
            return prepare_text(value, s.ragas_reference_chars)
        return value

    def _run(rows: list[dict], metrics) -> dict[str, list]:
        if not rows:
            return {}
        dataset = EvaluationDataset.from_list(
            [{k: _cap_field(k, r[k]) for k in _INPUT_FIELDS if k in r} for r in rows]
        )
        result = evaluate(
            dataset, metrics=metrics, llm=llm, embeddings=embeddings,
            run_config=run_config, show_progress=True, raise_exceptions=False,
        )
        df = result.to_pandas()
        metric_cols = [c for c in df.columns if c not in _INPUT_FIELDS]
        return {c: df[c].tolist() for c in metric_cols}

    # No-reference metrics over every sampled item.
    all_scores = _run(samples, [Faithfulness(), ResponseRelevancy()])
    # Reference-based metrics only where a ground-truth answer exists (the forum slice).
    ref_samples = [s for s in samples if s.get("reference")]
    ref_scores = _run(
        ref_samples,
        [AnswerCorrectness(), LLMContextPrecisionWithReference(), LLMContextRecall()],
    )

    # Attach per-item scores back onto the samples (aligned by row order).
    for i, s in enumerate(samples):
        s["scores"] = {m: vals[i] for m, vals in all_scores.items()}
    for j, s in enumerate(ref_samples):
        s["scores"].update({m: vals[j] for m, vals in ref_scores.items()})

    def _agg(rows: list[dict]) -> dict[str, dict]:
        # Report mean AND the number of items each metric actually scored (n): RAGAS emits
        # NaN when a judge call fails (413/parse), so mean-over-non-null with an explicit n
        # keeps a free-tier-degraded run honest instead of hiding failures in the average.
        keys = sorted({k for r in rows for k in r.get("scores", {})})
        out = {}
        for k in keys:
            vals = [r["scores"].get(k) for r in rows]
            n = sum(1 for v in vals if isinstance(v, (int | float)) and v == v)
            out[k] = {"mean": _mean(vals), "n": n}
        return out

    by_source: dict[str, dict] = {}
    for src in sorted({s["source"] for s in samples}):
        by_source[src] = _agg([s for s in samples if s["source"] == src])

    return {
        "metrics_overall": _agg(samples),
        "metrics_by_source": by_source,
        "n_with_reference": len(ref_samples),
    }


def _load_dataset(path: str) -> list[dict]:
    with _abs(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _save_dataset(samples: list[dict], stamp: str) -> Path:
    results_dir = _abs(get_settings().results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"ragas_dataset_{stamp}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGAS answer-quality eval (Layer 5d, D-040).")
    ap.add_argument("--sample", type=int, default=24, help="number of gold items to grade")
    ap.add_argument("--seed", type=int, default=13, help="sampling seed (reproducibility)")
    ap.add_argument("--gold", default=None, help="gold JSONL (default: config gold_jsonl)")
    ap.add_argument(
        "--from-dataset",
        default=None,
        help="score an existing ragas_dataset_*.jsonl instead of re-running the agent",
    )
    ap.add_argument("--no-save", action="store_true", help="print only; do not write results")
    ap.add_argument(
        "--synthesis-model",
        default=None,
        help="override the agent's synthesis model for this run (e.g. groq/llama-3.1-8b-instant "
        "when Gemini's 20/day free cap is spent); default: config synthesis_model (Gemini)",
    )
    args = ap.parse_args()

    s = get_settings()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if args.from_dataset:
        samples = _load_dataset(args.from_dataset)
        print(f"Loaded {len(samples)} cached samples from {args.from_dataset}")
        dataset_path = _abs(args.from_dataset)
    else:
        gold_file = args.gold or s.gold_jsonl
        items = stratified_sample(load_gold(gold_file), args.sample, args.seed)
        seeds_by_id = {seed.id: seed for seed in load_seeds()}
        synth = args.synthesis_model or s.synthesis_model
        print(f"Running agent over {len(items)} items (seed={args.seed}, synthesis={synth})...")
        samples = build_samples(items, seeds_by_id, synthesis_model=args.synthesis_model)
        dataset_path = None if args.no_save else _save_dataset(samples, stamp)

    # The synthesis model that produced these answers is stamped on the samples (so a
    # --from-dataset re-score reports the truth, not the current config default).
    synth_used = samples[0].get("synthesis_model") if samples else None
    synth_used = synth_used or s.synthesis_model

    print("\nScoring with RAGAS (judge routed through the gateway, Groq-8B)...")
    scored = score_samples(samples)

    results = {
        "layer": "5d",
        "eval": "ragas",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "judge_model": s.cheap_model,
        "embeddings_model": s.embedding_model,
        "sample": {
            "n": len(samples),
            "seed": args.seed,
            "gold": args.from_dataset or (args.gold or s.gold_jsonl),
            "by_source_counts": {
                src: sum(1 for x in samples if x["source"] == src)
                for src in sorted({x["source"] for x in samples})
            },
        },
        "agent": {
            "synthesis_model": synth_used,
            "synthesis_overridden": synth_used != s.synthesis_model,
            "verifier_model": s.cheap_model,
            "agent_context_k": s.agent_context_k,
            "agent_max_retries": s.agent_max_retries,
        },
        **scored,
    }

    print("\n=== RAGAS answer-quality (judge:", s.cheap_model, ") ===")
    print("overall:", json.dumps(results["metrics_overall"]))
    for src, d in results["metrics_by_source"].items():
        print(f"  {src}: {json.dumps(d)}")
    print(f"n_with_reference (forum, scored on correctness/context): {results['n_with_reference']}")

    if not args.no_save:
        results_dir = _abs(s.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"ragas_{stamp}.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved: {out_path}")
        if dataset_path:
            print(f"Dataset: {dataset_path}  (re-score with --from-dataset)")


if __name__ == "__main__":
    main()
