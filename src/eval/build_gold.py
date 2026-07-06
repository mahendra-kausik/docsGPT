"""Merge the real (forum) and synthetic slices into the final gold set (D-025).

The gold set is deliberately two-source (D-025): the answer-link forum questions
are *real* natural labels (the accepted answer linked that exact docs page), and
the synthetic questions give volume with docs-grounded gold. This step concatenates
the scored items from both into the single committed ``gold.jsonl`` that run_eval
reads, keeping each item's ``source`` tag so the baseline can report the two slices
separately (real is the honest headline; synthetic is augmentation).

Run:  ./tasks.ps1 build-gold     (after compile-gold + synth)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.config import PROJECT_ROOT, get_settings
from src.eval.gold import GoldItem, load_gold, save_gold


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_if_exists(path: str) -> list[GoldItem]:
    """Load a gold slice, or return [] if that slice hasn't been produced yet."""
    return load_gold(path) if _abs(path).exists() else []


def build(forum_path: str, synth_path: str) -> tuple[list[GoldItem], dict]:
    """Merge the scored items of both slices; return (items, report)."""
    forum = [it for it in _load_if_exists(forum_path) if it.is_scored]
    synth = [it for it in _load_if_exists(synth_path) if it.is_scored]
    items = forum + synth
    report = {
        "total": len(items),
        "by_source": dict(Counter(it.source for it in items)),
        "by_hop": dict(Counter(it.hop for it in items if it.hop)),
    }
    return items, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge forum + synthetic slices into gold.jsonl.")
    ap.add_argument("--forum", default=None, help="forum gold slice (default from config)")
    ap.add_argument("--synth", default=None, help="synthetic gold slice (default from config)")
    ap.add_argument("--out", default=None, help="merged output (default from config)")
    args = ap.parse_args()
    s = get_settings()

    items, report = build(args.forum or s.gold_forum_jsonl, args.synth or s.gold_synth_jsonl)
    if not items:
        raise SystemExit(
            "No scored gold items found. Produce the slices first: "
            "./tasks.ps1 compile-gold  and/or  ./tasks.ps1 synth"
        )
    path = save_gold(items, args.out or s.gold_jsonl)
    print(json.dumps({"gold_file": str(path), **report}, indent=2))


if __name__ == "__main__":
    main()
