"""Candidate-proposal tool for building the gold set (Layer 3, D-024).

Batch workflow (chosen over interactive review to keep it fast and offline):
for every forum seed, run the dense retriever, pull the accepted-answer text from
the raw cache, and emit two files:

- ``candidates.jsonl`` (machine): per question, the ordered candidate chunk ids —
  the source of truth the compiler uses to map a reviewed rank back to a chunk id.
- ``review.md`` (human): the question, its accepted forum answer, and the numbered
  candidate docs chunks, each with a DECISION line the user fills in one pass.

The candidate pool (``eval_candidate_pool``, default 20) is deliberately deeper
than the reported cutoffs (@1/@5/@10) so the gold mapping isn't capped at k=10 —
this reduces (does not eliminate) the bias of grading the retriever against gold
that the same retriever surfaced (documented in D-024).

Run:  ./tasks.ps1 propose        (needs the live Qdrant cluster)
Then edit review.md, then:  ./tasks.ps1 compile-gold
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pydantic import BaseModel

from src.config import PROJECT_ROOT, get_settings
from src.eval.gold import ForumSeed, accepted_answer_text, load_seeds
from src.retrieval.search import DenseRetriever

logger = logging.getLogger(__name__)


class Candidate(BaseModel):
    """One proposed docs chunk for a question, with just enough to review + compile."""

    rank: int
    chunk_id: str
    score: float
    heading_path: str
    source_url: str
    snippet: str


class Proposal(BaseModel):
    """A question plus its ranked candidate chunks — one record in candidates.jsonl."""

    qid: int
    question: str
    source_url: str
    tags: list[str] = []
    candidates: list[Candidate] = []


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _snippet(text: str, n: int) -> str:
    """Collapse whitespace and cap length for a compact one-line preview."""
    return " ".join(text.split())[:n]


def build_proposal(
    seed: ForumSeed, retriever: DenseRetriever, pool: int, snippet_chars: int
) -> Proposal:
    """Retrieve the candidate pool for one seed's question."""
    hits, _ = retriever.search(seed.question, top_k=pool)
    candidates = [
        Candidate(
            rank=i,
            chunk_id=h.id,
            score=round(h.score, 4),
            heading_path=h.heading_path,
            source_url=h.source_url,
            snippet=_snippet(h.text, snippet_chars),
        )
        for i, h in enumerate(hits, start=1)
    ]
    return Proposal(
        qid=seed.id,
        question=seed.question,
        source_url=seed.url,
        tags=seed.tags,
        candidates=candidates,
    )


REVIEW_HEADER = """# Gold-set review — map each forum question to its answering docs chunk(s)

**How to review** — edit each `DECISION` line, save, then run `./tasks.ps1 compile-gold`:

- Read the **question** + its **accepted forum answer**, then the numbered **candidate chunks**.
- On `DECISION`, list the candidate **number(s)** whose chunk answers it, e.g. `1` or `2 5`.
- If **none** fit (a docs-corpus gap), write `x` — it becomes *unanswerable* and is dropped
  from the scored set (kept only as a coverage stat).
- Know a correct chunk that isn't listed? Paste its raw `chunk_id` (hex) alongside any numbers.
- Leave a `DECISION` line **blank** to mark it *pending* — the compiler skips it and warns.

Total questions: {n}

---

"""


def render_review(
    proposals: list[Proposal], seeds_by_id: dict[int, ForumSeed], snippet_chars: int
) -> str:
    """Render the human-facing review.md from the proposals + accepted answers."""
    parts = [REVIEW_HEADER.format(n=len(proposals))]
    for prop in proposals:
        seed = seeds_by_id[prop.qid]
        answer = _snippet(accepted_answer_text(seed), 1500) or "(raw answer not cached)"
        tags = ", ".join(prop.tags) or "—"
        parts.append(f"## [{prop.qid}] {seed.title}\n")
        parts.append(f"- Forum: {prop.source_url}  |  tags: {tags}\n")
        parts.append(f"\n**Question:**\n\n{_snippet(prop.question, 1200)}\n")
        parts.append(f"\n**Accepted forum answer:**\n\n{answer}\n")
        parts.append(f"\n**Candidate docs chunks (dense top-{len(prop.candidates)}):**\n")
        for c in prop.candidates:
            parts.append(
                f"{c.rank}. `{c.score:.4f}`  {c.heading_path or '(no heading)'}\n"
                f"   {c.source_url}\n"
                f"   {c.snippet}\n"
            )
        parts.append(f"\n> DECISION [qid={prop.qid}]: \n\n---\n")
    return "\n".join(parts)


def _load_candidates(path: Path) -> list[Proposal]:
    """Read existing proposals back from candidates.jsonl (for offline re-render)."""
    with path.open("r", encoding="utf-8") as fh:
        return [Proposal.model_validate_json(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Propose gold candidates + write review.md.")
    ap.add_argument(
        "--from-candidates",
        action="store_true",
        help="re-render review.md from the existing candidates.jsonl (no cluster query)",
    )
    args = ap.parse_args()
    s = get_settings()
    seeds = load_seeds()
    cand_path = _abs(s.gold_candidates_jsonl)

    if args.from_candidates:
        proposals = _load_candidates(cand_path)
        logger.info("re-rendering review.md from %d cached proposals", len(proposals))
    else:
        logger.info(
            "proposing candidates for %d forum seeds (pool=%d)", len(seeds), s.eval_candidate_pool
        )
        retriever = DenseRetriever()
        proposals = []
        for i, seed in enumerate(seeds, start=1):
            proposals.append(
                build_proposal(seed, retriever, s.eval_candidate_pool, s.eval_snippet_chars)
            )
            if i % 20 == 0:
                logger.info("  proposed %d/%d", i, len(seeds))
        cand_path.parent.mkdir(parents=True, exist_ok=True)
        with cand_path.open("w", encoding="utf-8") as fh:
            for prop in proposals:
                fh.write(prop.model_dump_json() + "\n")

    seeds_by_id = {seed.id: seed for seed in seeds}
    review_path = _abs(s.gold_review_md)
    review_path.write_text(
        render_review(proposals, seeds_by_id, s.eval_snippet_chars), encoding="utf-8"
    )

    print(f"Wrote review for {len(proposals)} proposals: {review_path}")
    if not args.from_candidates:
        print(f"  machine : {cand_path}")
    print("Next: ./tasks.ps1 prefill  (auto-suggest high-confidence labels), then edit review.md")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
