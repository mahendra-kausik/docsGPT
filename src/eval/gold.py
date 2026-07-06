"""Gold-set data model + forum-seed loading for the eval harness (Layer 3).

The gold set is the reference every later layer is measured against (PLAN §3/§6).
Each item pairs a *real* forum question (Layer 1b seed) with the MIT-docs chunk(s)
that answer it. Mapping is human-verified via the propose -> review -> compile
workflow (D-024); questions not answerable from the docs corpus are marked
``unanswerable`` and dropped from the scored set, kept only as a coverage stat
(D-019, and the Layer 3 unanswerable policy).

This module holds the schemas + JSONL/raw-cache I/O; scoring lives in metrics.py.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from pydantic import BaseModel

from src.config import PROJECT_ROOT, get_settings

# Status values for a gold item after human review.
VERIFIED = "verified"  # >=1 gold chunk hand-confirmed to answer the question
UNANSWERABLE = "unanswerable"  # real question, but the docs corpus can't answer it (dropped)


class ForumSeed(BaseModel):
    """One solved forum topic as committed to data/gold/forum_seeds.jsonl (Layer 1b).

    Carries the question + metadata + accepted-answer URL, but no answer body — the
    body lives only in the git-ignored raw cache (D-019) and is read at review time.
    """

    id: int
    title: str
    url: str
    category: str = ""
    tags: list[str] = []
    created_at: str = ""
    question: str = ""
    accepted_answer_post: int | None = None
    accepted_answer_url: str = ""


class GoldItem(BaseModel):
    """A verified (or explicitly unanswerable) question -> gold-chunk mapping."""

    qid: int  # forum topic id (stable, traces back to the seed)
    question: str  # the user's real question (from the seed)
    gold_chunk_ids: list[str] = []  # docs chunk ids that answer it (empty if unanswerable)
    status: str = VERIFIED
    source_url: str = ""  # forum topic URL, for provenance/attribution
    tags: list[str] = []

    @property
    def is_scored(self) -> bool:
        """Only verified items with at least one gold chunk enter the metrics."""
        return self.status == VERIFIED and len(self.gold_chunk_ids) > 0


def _abs(path: str) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_seeds(path: str | None = None) -> list[ForumSeed]:
    """Load the committed forum gold seeds (question + metadata, no answer body)."""
    s = get_settings()
    seeds_path = _abs(path or s.forum.seeds_jsonl)
    with seeds_path.open("r", encoding="utf-8") as fh:
        return [ForumSeed.model_validate_json(line) for line in fh if line.strip()]


# --- Accepted-answer text extraction from the raw Discourse cache (review-time only) ---

_BLOCK_BREAK = re.compile(r"</(p|div|li|pre|h[1-6]|blockquote|tr)>", re.IGNORECASE)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_MULTI_NL = re.compile(r"\n{3,}")


def html_to_text(cooked: str) -> str:
    """Flatten Discourse ``cooked`` HTML to readable plain text.

    Kept dependency-free (stdlib only, per D-015): insert newlines at block
    boundaries, drop remaining tags, unescape entities. Good enough for a human
    to read an accepted answer beside the retrieved candidates — not a full parse.
    """
    text = _BR.sub("\n", cooked)
    text = _BLOCK_BREAK.sub("\n", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _accepted_post(seed: ForumSeed, raw_dir: str | None = None) -> dict:
    """Return the accepted answer's raw post dict from the cache ({} if uncached).

    Matches the post whose ``post_number`` equals the seed's ``accepted_answer_post``;
    falls back to the first post flagged ``accepted_answer``.
    """
    s = get_settings()
    topic_path = _abs(raw_dir or s.forum.raw_dir) / "topics" / f"{seed.id}.json"
    if not topic_path.exists():
        return {}
    topic = json.loads(topic_path.read_text(encoding="utf-8"))
    posts = (topic.get("post_stream") or {}).get("posts") or []

    chosen = None
    if seed.accepted_answer_post is not None:
        chosen = next((p for p in posts if p.get("post_number") == seed.accepted_answer_post), None)
    if chosen is None:
        chosen = next((p for p in posts if p.get("accepted_answer")), None)
    return chosen or {}


def accepted_answer_html(seed: ForumSeed, raw_dir: str | None = None) -> str:
    """Raw ``cooked`` HTML of the accepted answer (keeps <a href> links, unlike text)."""
    return _accepted_post(seed, raw_dir).get("cooked", "")


def accepted_answer_text(seed: ForumSeed, raw_dir: str | None = None) -> str:
    """Readable plain text of the accepted answer ("" if the raw topic isn't cached)."""
    return html_to_text(accepted_answer_html(seed, raw_dir))


# --- Gold set JSONL I/O ---


def load_gold(path: str) -> list[GoldItem]:
    """Load a verified gold set from JSONL."""
    gold_path = _abs(path)
    with gold_path.open("r", encoding="utf-8") as fh:
        return [GoldItem.model_validate_json(line) for line in fh if line.strip()]


def save_gold(items: list[GoldItem], path: str) -> Path:
    """Write the gold set to JSONL (one item per line); returns the resolved path."""
    gold_path = _abs(path)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with gold_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(item.model_dump_json() + "\n")
    return gold_path
