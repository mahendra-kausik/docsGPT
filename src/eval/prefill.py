"""Pre-fill high-confidence gold labels from answer-link URL matches (Layer 3, D-024).

The strongest natural signal for a question's gold chunk is not the dense score —
neighboring chunks from one doc page score near-identically (gap-to-#2 p75 ≈ 0.02),
so score alone can't separate them — but the fact that **the accepted forum answer
explicitly links a docs page** that is also in the candidate pool. Those are
high-precision labels the reviewer only needs to *spot-check*, not build from scratch.

This tool matches each accepted answer's `docs.langchain.com` links against the
candidate chunks' `source_url` (normalizing away the `/oss/python|js/` language
segment the docs site adds but our chunk URLs omit), then fills the matched
candidate rank(s) into the corresponding **blank** DECISION line in review.md —
never overwriting a line you've already edited — and annotates it so you know it
was auto-suggested. Everything remains yours to confirm or change before compile.

Run:  ./tasks.ps1 prefill        (after ./tasks.ps1 propose)
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from src.config import PROJECT_ROOT, get_settings
from src.eval.compile_gold import load_proposals
from src.eval.gold import ForumSeed, accepted_answer_html, load_seeds
from src.eval.propose import Proposal

logger = logging.getLogger(__name__)

_HREF = re.compile(r'href="(https?://docs\.langchain\.com[^"]*)"', re.IGNORECASE)
_LANG_SEG = re.compile(r"/oss/(?:python|js|javascript)/")
_ANNOTATION = "> _auto-suggested from an answer link — verify before compiling_"


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def normalize_docs_url(url: str) -> str:
    """Canonicalize a docs URL for matching: drop scheme, #anchor/?query, and the
    `/oss/python|js/` language segment (present in rendered links, absent from our
    chunk URLs), then trailing slash + case."""
    u = re.sub(r"^https?://", "", url or "")
    u = re.sub(r"[#?].*$", "", u)
    u = _LANG_SEG.sub("/oss/", u)
    return u.rstrip("/").lower()


def answer_link_matches(seed: ForumSeed, prop: Proposal) -> list[int]:
    """Ranks of the best chunk of each docs page the accepted answer links.

    A linked page is often split across several pooled chunks sharing one
    ``source_url``; marking them all as gold would inflate recall (any of N counts
    as a hit). So we keep only the **best-ranked (lowest-rank) chunk per distinct
    linked page** — one clean representative per page — which the reviewer can
    still extend by hand. Returns ranks sorted ascending.
    """
    links = {normalize_docs_url(m.group(1)) for m in _HREF.finditer(accepted_answer_html(seed))}
    if not links:
        return []
    best_rank_by_page: dict[str, int] = {}
    for c in prop.candidates:  # candidates are already best-first
        page = normalize_docs_url(c.source_url)
        if page in links and page not in best_rank_by_page:
            best_rank_by_page[page] = c.rank
    return sorted(best_rank_by_page.values())


def apply_prefill(review_md: str, prefill: dict[int, list[int]]) -> tuple[str, int]:
    """Fill blank DECISION lines for matched qids; return (new_md, n_filled).

    Idempotent: a DECISION line that already has any content is left untouched, and
    the annotation is only inserted once.
    """
    out: list[str] = []
    filled = 0
    prev_annotated = False
    decision_re = re.compile(r"^(> DECISION \[qid=(\d+)\]:)(\s*)(.*)$")
    for line in review_md.splitlines():
        m = decision_re.match(line)
        if not m:
            out.append(line)
            prev_annotated = line.strip() == _ANNOTATION.strip()
            continue
        prefix, qid_s, _, existing = m.groups()
        qid = int(qid_s)
        if existing.strip() or qid not in prefill:
            out.append(line)  # already decided, or nothing to suggest
            continue
        ranks = " ".join(str(r) for r in prefill[qid])
        if not prev_annotated:
            out.append(_ANNOTATION)
        out.append(f"{prefix} {ranks}")
        filled += 1
        prev_annotated = False
    return "\n".join(out) + "\n", filled


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-fill gold labels from answer-link URL matches.")
    ap.add_argument("--candidates", default=None, help="candidates.jsonl (default from config)")
    ap.add_argument("--review", default=None, help="review.md to edit in place (default: config)")
    args = ap.parse_args()
    s = get_settings()

    proposals = load_proposals(args.candidates or s.gold_candidates_jsonl)
    seeds_by_id = {seed.id: seed for seed in load_seeds()}

    prefill: dict[int, list[int]] = {}
    for qid, prop in proposals.items():
        seed = seeds_by_id.get(qid)
        if seed is None:
            continue
        ranks = answer_link_matches(seed, prop)
        if ranks:
            prefill[qid] = ranks

    review_path = _abs(args.review or s.gold_review_md)
    new_md, filled = apply_prefill(review_path.read_text(encoding="utf-8"), prefill)
    review_path.write_text(new_md, encoding="utf-8")

    print(f"Answer-link matches found for {len(prefill)} / {len(proposals)} questions.")
    print(f"Filled {filled} previously-blank DECISION line(s) in {review_path.name}.")
    print("These are SUGGESTIONS - open review.md, verify each annotated line, then compile-gold.")
    if len(prefill) != filled:
        skipped = len(prefill) - filled
        print(f"({skipped} match(es) skipped: those DECISION lines were already edited.)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
