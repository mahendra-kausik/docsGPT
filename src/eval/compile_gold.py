"""Compile the human-reviewed decisions into a verified gold set (Layer 3, D-024).

Reads the DECISION lines from ``review.md`` and maps each accepted candidate
number back to a chunk id via ``candidates.jsonl`` (the machine source of truth),
producing ``gold.jsonl``. Verified items (>=1 gold chunk) are what the baseline
eval scores; ``x`` items are recorded as ``unanswerable`` (a docs-corpus coverage
stat, dropped from metrics per D-019); blank decisions are reported as pending.

Run:  ./tasks.ps1 compile-gold
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from src.config import PROJECT_ROOT, get_settings
from src.eval.gold import UNANSWERABLE, VERIFIED, GoldItem
from src.eval.propose import Proposal

logger = logging.getLogger(__name__)

# One decision per question, e.g. ``> DECISION [qid=4050]: 1 3``
_DECISION = re.compile(r"^>\s*DECISION\s*\[qid=(\d+)\]:\s*(.*)$")
_UNANSWERABLE_TOKENS = {"x", "none", "skip", "-", "n/a", "na"}
_CHUNK_ID = re.compile(r"^[0-9a-f]{8,}$")  # literal chunk id (hex hash, D-021)


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_proposals(path: str) -> dict[int, Proposal]:
    """Load candidates.jsonl keyed by qid (rank -> chunk_id mapping lives here)."""
    proposals: dict[int, Proposal] = {}
    with _abs(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                prop = Proposal.model_validate_json(line)
                proposals[prop.qid] = prop
    return proposals


def parse_decisions(review_md: str) -> dict[int, str]:
    """Extract {qid: raw decision string} from every DECISION line in review.md."""
    decisions: dict[int, str] = {}
    for line in review_md.splitlines():
        m = _DECISION.match(line.strip())
        if m:
            decisions[int(m.group(1))] = m.group(2).strip()
    return decisions


def resolve_gold_ids(raw: str, prop: Proposal) -> list[str]:
    """Turn a decision string into concrete chunk ids.

    Integers select candidates by their shown rank; hex tokens are taken as
    literal chunk ids (for the case where the reviewer knew a chunk not in the
    pool). Order is preserved and duplicates removed. Unknown ranks raise so a
    typo surfaces loudly rather than silently dropping a label.
    """
    by_rank = {c.rank: c.chunk_id for c in prop.candidates}
    ids: list[str] = []
    for tok in raw.split():
        if tok.isdigit():
            rank = int(tok)
            if rank not in by_rank:
                raise ValueError(
                    f"qid={prop.qid}: candidate #{rank} out of range (pool={len(by_rank)})"
                )
            cid = by_rank[rank]
        elif _CHUNK_ID.match(tok.lower()):
            cid = tok.lower()
        else:
            raise ValueError(f"qid={prop.qid}: unrecognized decision token {tok!r}")
        if cid not in ids:
            ids.append(cid)
    return ids


def compile_gold(
    proposals: dict[int, Proposal], decisions: dict[int, str]
) -> tuple[list[GoldItem], dict[str, int]]:
    """Build gold items from proposals + decisions; return (items, counts)."""
    items: list[GoldItem] = []
    counts = {"verified": 0, "unanswerable": 0, "pending": 0}
    for qid, prop in proposals.items():
        raw = decisions.get(qid, "").strip()
        if not raw:
            counts["pending"] += 1
            logger.warning("qid=%d: no decision yet (pending) - skipped", qid)
            continue
        if raw.lower() in _UNANSWERABLE_TOKENS:
            counts["unanswerable"] += 1
            items.append(
                GoldItem(
                    qid=qid,
                    question=prop.question,
                    gold_chunk_ids=[],
                    status=UNANSWERABLE,
                    source_url=prop.source_url,
                    tags=prop.tags,
                )
            )
            continue
        gold_ids = resolve_gold_ids(raw, prop)
        counts["verified"] += 1
        items.append(
            GoldItem(
                qid=qid,
                question=prop.question,
                gold_chunk_ids=gold_ids,
                status=VERIFIED,
                source_url=prop.source_url,
                tags=prop.tags,
            )
        )
    return items, counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile reviewed decisions into gold.jsonl")
    ap.add_argument(
        "--candidates", default=None, help="candidates.jsonl path (default from config)"
    )
    ap.add_argument("--review", default=None, help="review.md path (default from config)")
    ap.add_argument("--out", default=None, help="output forum-gold path (default from config)")
    args = ap.parse_args()
    s = get_settings()

    proposals = load_proposals(args.candidates or s.gold_candidates_jsonl)
    review_md = _abs(args.review or s.gold_review_md).read_text(encoding="utf-8")
    decisions = parse_decisions(review_md)

    items, counts = compile_gold(proposals, decisions)

    out_path = _abs(args.out or s.gold_forum_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")

    scored = sum(1 for it in items if it.is_scored)
    print(
        json.dumps(
            {
                "gold_file": str(out_path),
                "total_questions": len(proposals),
                "verified": counts["verified"],
                "unanswerable_dropped": counts["unanswerable"],
                "pending_skipped": counts["pending"],
                "scored_items": scored,
            },
            indent=2,
        )
    )
    if counts["pending"]:
        print(
            f"\n[!] {counts['pending']} question(s) still have a blank DECISION line - "
            "fill them in review.md and re-run to include them."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
