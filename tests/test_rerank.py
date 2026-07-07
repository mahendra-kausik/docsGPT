"""Unit tests for the pure rerank-ordering helper (Layer 4b, D-029).

Offline/pure like the other retrieval tests: the cross-encoder itself needs torch +
a model download, exercised by the eval gate, not here. What's pinned here is the
reordering contract — reranker scores replace retrieval scores, order is by score
desc with deterministic id tie-breaks, and top_k caps the output.
"""

from src.retrieval.rerank import apply_rerank
from src.retrieval.search import Hit


def _hit(cid: str, score: float = 0.0) -> Hit:
    return Hit(id=cid, score=score, text=f"text-{cid}", source_url="", heading_path="")


def test_reorders_by_rerank_score_and_replaces_score():
    hits = [_hit("a"), _hit("b"), _hit("c")]
    out = apply_rerank(hits, [0.1, 0.9, 0.5], top_k=3)
    assert [h.id for h in out] == ["b", "c", "a"]
    assert out[0].score == 0.9  # retrieval score is replaced by the rerank score


def test_top_k_caps_the_output():
    hits = [_hit("a"), _hit("b"), _hit("c")]
    out = apply_rerank(hits, [0.1, 0.9, 0.5], top_k=2)
    assert [h.id for h in out] == ["b", "c"]


def test_ties_break_deterministically_by_id():
    hits = [_hit("b"), _hit("a")]
    out = apply_rerank(hits, [0.5, 0.5], top_k=2)
    assert [h.id for h in out] == ["a", "b"]  # equal score -> id-sorted, reproducible


def test_empty_input_yields_empty_output():
    assert apply_rerank([], [], top_k=5) == []
