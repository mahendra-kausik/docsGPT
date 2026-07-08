"""Measure the agent's per-query cost: LLM calls, tokens, p50/p95 latency (Layer 6, PLAN §6).

Runs the in-process agent over a list of questions inside `collect_metrics()` and prints the
distribution. Reuses the gateway's on-disk cache, so re-running a fixed question set is FREE and
reproducible (cached calls spend no quota and are reported as cache hits). Uncached questions DO
make live LLM calls — mind the free-tier caps (Gemini ~20/day) before running a large fresh set.

Run:  ./tasks.ps1 bench                      # built-in sample questions
      ./tasks.ps1 bench --questions q.txt    # one question per line
"""

from __future__ import annotations

import argparse
import time

from src.agent.graph import answer_question
from src.llm.metrics import collect_metrics

# Small default set for a quick, mostly-cached read. Override with --questions for a real batch.
_DEFAULT_QUESTIONS = [
    "How do I add message history to a chain?",
    "How do I stream tokens from a chat model?",
    "How do I create a tool in LangChain?",
]


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation) — honest for the small n we run here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark agent cost/latency per query (Layer 6).")
    ap.add_argument("--questions", help="path to a file with one question per line")
    ap.add_argument("--max-retries", type=int, default=None, help="self-correction budget override")
    args = ap.parse_args()

    if args.questions:
        with open(args.questions, encoding="utf-8") as fh:
            questions = [ln.strip() for ln in fh if ln.strip()]
    else:
        questions = _DEFAULT_QUESTIONS

    latencies: list[float] = []
    calls: list[int] = []
    tokens: list[int] = []
    cache_hits = 0

    for q in questions:
        t0 = time.perf_counter()
        with collect_metrics() as m:
            state = answer_question(q, max_retries=args.max_retries)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)
        calls.append(m.calls)
        tokens.append(m.total_tokens)
        cache_hits += m.cache_hits
        grounded = state.get("grounded", True)
        print(
            f"  {dt_ms:8.1f} ms | calls={m.calls} cache_hits={m.cache_hits} "
            f"tokens={m.total_tokens:5d} grounded={grounded} | {q[:48]}"
        )

    n = len(questions)
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    print(f"\nBenchmark over {n} question(s):")
    print(f"  latency  p50={p50:.1f} ms  p95={p95:.1f} ms")
    print(f"  LLM calls/query   mean={_mean(calls):.2f}")
    print(f"  tokens/query      mean={_mean(tokens):.1f}")
    print(f"  cache hits total  {cache_hits}")


if __name__ == "__main__":
    main()
