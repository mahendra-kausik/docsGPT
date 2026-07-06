"""Synthetic gold generation: Groq-8B questions grounded in docs chunks (D-025).

Most forum answers don't point at a single docs chunk (114/163 don't link docs at
all), so hand-mapping them into retrieval gold is neither feasible nor honest. The
rebalance (D-025) makes the bulk of the gold set **synthetic questions generated
from the corpus**: sample a chunk (or a few chunks of one page), have Groq-8B write
a developer question answerable from them, and take those chunk ids as gold — known
by construction, so no human relevance-judging and the questions stay docs-grounded
and readable. The 26 answer-link forum questions remain as a real-question slice.

Determinism: chunk selection is seeded and every LLM call is cached (src/llm), so a
re-run reproduces the same questions; the committed gold file is the reproducible
artifact (CLAUDE.md §4), not the (one-time, non-deterministic) generation itself.

Run:  ./tasks.ps1 synth -- --n-single 7 --n-multi 3     (small sample to eyeball)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from src.config import PROJECT_ROOT, get_settings
from src.eval.gold import GoldItem, save_gold
from src.ingest.models import Chunk
from src.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)

SYNTH_QID_BASE = 1_000_000  # keep synthetic ids clear of forum topic ids (< ~5000)

_SINGLE_SYS = (
    "You write realistic questions a developer would ask a documentation Q&A "
    "assistant. Given ONE documentation passage, write a single, specific question "
    "that the passage fully answers. Rules: the question must be self-contained and "
    "natural (never say 'the passage', 'the text', 'this document', or 'above'); it "
    "must be answerable from the passage alone; prefer concrete API/behavior details. "
    'Respond with strict JSON: {"question": "..."}.'
)
_MULTI_SYS = (
    "You write realistic multi-part questions a developer would ask a documentation "
    "Q&A assistant. Given SEVERAL passages from one documentation page, write a single "
    "question that genuinely requires information from ALL of them to answer fully. "
    "Rules: natural and self-contained (never mention 'passages'/'text'/'document'); "
    'answerable only by combining them. Respond with strict JSON: {"question": "..."}.'
)

# Reject a generated question that leaked meta-references or is too short/long.
_LEAK_TERMS = ("passage", "the text", "this document", "the document", "above", "the snippet")


def load_chunks(path: str | None = None) -> list[Chunk]:
    """Read the committed corpus chunks (the durable source of truth, D-004)."""
    s = get_settings()
    p = Path(path or s.corpus_jsonl)
    p = p if p.is_absolute() else PROJECT_ROOT / p
    with p.open("r", encoding="utf-8") as fh:
        return [Chunk.model_validate_json(line) for line in fh if line.strip()]


def _clean_question(raw: str) -> str | None:
    """Parse the model's JSON, validate the question, or return None to skip it."""
    try:
        q = json.loads(raw).get("question", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return None
    if not (15 <= len(q) <= 300) or not q.endswith("?"):
        return None
    low = q.lower()
    if any(term in low for term in _LEAK_TERMS):
        return None
    return q


def _passage(chunk: Chunk) -> str:
    """Format one chunk (heading breadcrumb + text) for the prompt."""
    head = chunk.heading_path or chunk.title or ""
    return f"[{head}]\n{chunk.text}" if head else chunk.text


def generate_single(
    chunks: list[Chunk], n: int, gw: LLMGateway, rng: random.Random
) -> list[GoldItem]:
    """One question per sampled chunk; gold = that chunk id."""
    pool = [c for c in chunks if c.n_chars >= gw.s.synth_min_chunk_chars]
    rng.shuffle(pool)
    items: list[GoldItem] = []
    for chunk in pool:
        if len(items) >= n:
            break
        raw = gw.complete(_SINGLE_SYS, _passage(chunk), response_json=True, max_tokens=200)
        q = _clean_question(raw)
        if q is None:
            continue
        items.append(
            GoldItem(
                qid=SYNTH_QID_BASE + len(items),
                question=q,
                gold_chunk_ids=[chunk.id],
                source="synthetic",
                hop="single",
                source_url=chunk.source_url,
            )
        )
    return items


def generate_multi(
    chunks: list[Chunk], n: int, gw: LLMGateway, rng: random.Random, start_qid: int
) -> list[GoldItem]:
    """One question per sampled multi-chunk group from a single page; gold = those ids."""
    group_size = gw.s.synth_multi_group_size
    pages: dict[str, list[Chunk]] = {}
    for c in chunks:
        if c.n_chars >= gw.s.synth_min_chunk_chars and c.source_url:
            pages.setdefault(c.source_url, []).append(c)
    multi_pages = [cs for cs in pages.values() if len(cs) >= 2]
    rng.shuffle(multi_pages)

    items: list[GoldItem] = []
    for page_chunks in multi_pages:
        if len(items) >= n:
            break
        group = rng.sample(page_chunks, min(group_size, len(page_chunks)))
        body = "\n\n---\n\n".join(_passage(c) for c in group)
        raw = gw.complete(_MULTI_SYS, body, response_json=True, max_tokens=200)
        q = _clean_question(raw)
        if q is None:
            continue
        items.append(
            GoldItem(
                qid=start_qid + len(items),
                question=q,
                gold_chunk_ids=[c.id for c in group],
                source="synthetic",
                hop="multi",
                source_url=group[0].source_url,
            )
        )
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic gold questions from docs chunks.")
    ap.add_argument("--n-single", type=int, default=70, help="single-hop questions to generate")
    ap.add_argument("--n-multi", type=int, default=30, help="multi-hop questions to generate")
    ap.add_argument("--out", default=None, help="output jsonl (default: config gold_synth_jsonl)")
    args = ap.parse_args()
    s = get_settings()

    chunks = load_chunks()
    gw = LLMGateway()  # defaults to the cheap Groq model (llama-3.1-8b-instant)
    rng = random.Random(s.synth_seed)

    logger.info("generating %d single-hop + %d multi-hop from %d chunks",
                args.n_single, args.n_multi, len(chunks))
    single = generate_single(chunks, args.n_single, gw, rng)
    multi = generate_multi(chunks, args.n_multi, gw, rng, SYNTH_QID_BASE + len(single))
    items = single + multi

    out = args.out or s.gold_synth_jsonl
    path = save_gold(items, out)
    print(f"Generated {len(single)} single-hop + {len(multi)} multi-hop = {len(items)} items")
    print(f"Saved: {path}")
    print("Eyeball a few (question + gold heading), then we scale up and merge with the 26 real.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
