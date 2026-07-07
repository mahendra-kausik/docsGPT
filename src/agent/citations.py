"""Parse and validate inline [n] citations against the retrieved chunks (Layer 5a, D-034).

The synthesis node is told to cite claims with [n] markers indexing the numbered
context chunks. Keeping the parse/validate step pure (no LLM) makes the "every
citation resolves to a real chunk" guarantee cheap and unit-testable; whether a
claim is actually *supported* by its cited chunk is the deeper grounding check that
Layer 5c adds. Out-of-range markers are surfaced as hallucinated citations, not
silently dropped, so the agent (5c) can react to them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.search import Hit

# One or more integers (comma/space separated) inside square brackets: [1], [1, 3].
# Letters inside the brackets (e.g. "[binary omitted]") fail the class, so prose and
# markdown link labels are never mistaken for citations.
_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Citation:
    """A resolved citation: the [n] marker and the source it points to."""

    marker: int
    chunk_id: str
    source_url: str
    heading_path: str


def parse_markers(text: str) -> list[int]:
    """Return the cited numbers in first-appearance order, de-duplicated."""
    seen: dict[int, None] = {}  # dict preserves insertion order and dedupes
    for group in _MARKER.findall(text):
        for part in group.split(","):
            seen.setdefault(int(part.strip()), None)
    return list(seen)


def resolve_citations(
    text: str, chunks: list[Hit]
) -> tuple[list[Citation], list[int]]:
    """Map [n] markers to chunks (1-based). Returns (resolved, invalid_markers)."""
    resolved: list[Citation] = []
    invalid: list[int] = []
    for m in parse_markers(text):
        if 1 <= m <= len(chunks):
            c = chunks[m - 1]
            resolved.append(Citation(m, c.id, c.source_url, c.heading_path))
        else:
            invalid.append(m)
    return resolved, invalid
