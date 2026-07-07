"""Unit tests for citation parsing/validation (Layer 5a, D-034).

Pure + offline: the synthesis node emits inline [n] markers indexing the retrieved
chunks; this layer resolves them to real sources and flags any out-of-range marker
(a hallucinated citation). Pinned here so the "citations resolve to real chunks" gate
needs no LLM.
"""

from src.agent.citations import parse_markers, resolve_citations
from src.retrieval.search import Hit


def _chunk(cid: str) -> Hit:
    return Hit(id=cid, score=0.0, text=f"text-{cid}", source_url=f"url-{cid}", heading_path=cid)


def test_parse_simple_markers_in_order_unique():
    assert parse_markers("A [1] then B [2] and again [1].") == [1, 2]


def test_parse_comma_and_adjacent_forms():
    assert parse_markers("multi [1, 3] and adjacent [2][4]") == [1, 3, 2, 4]


def test_parse_ignores_non_numeric_brackets():
    # e.g. a stray "[binary omitted]" or markdown link text must not be read as a cite
    assert parse_markers("see [binary omitted] and [note] here [2]") == [2]


def test_resolve_maps_valid_markers_to_sources():
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    cites, invalid = resolve_citations("claim one [1]. claim two [3].", chunks)
    assert invalid == []
    assert [c.marker for c in cites] == [1, 3]
    assert [c.chunk_id for c in cites] == ["a", "c"]
    assert cites[0].source_url == "url-a"


def test_resolve_flags_out_of_range_markers_as_invalid():
    chunks = [_chunk("a"), _chunk("b")]
    cites, invalid = resolve_citations("good [1], hallucinated [5].", chunks)
    assert [c.marker for c in cites] == [1]
    assert invalid == [5]


def test_resolve_dedupes_repeated_markers():
    chunks = [_chunk("a"), _chunk("b")]
    cites, invalid = resolve_citations("[1] and again [1] and [2]", chunks)
    assert [c.marker for c in cites] == [1, 2]
    assert invalid == []


def test_resolve_empty_when_no_markers():
    assert resolve_citations("no citations here", [_chunk("a")]) == ([], [])
