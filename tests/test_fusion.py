"""Unit tests for client-side Reciprocal Rank Fusion (Layer 4a, D-027).

Pure + offline: RRF is the headline of the dense-vs-hybrid ablation, so its ranking
math is pinned here (needs neither fastembed nor a live cluster). score(d) =
sum over lists of 1/(k + rank), rank 1-based.
"""

from src.retrieval.fusion import rrf_fuse


def test_single_list_preserves_order():
    fused = rrf_fuse([["a", "b", "c"]], k=60)
    assert [i for i, _ in fused] == ["a", "b", "c"]


def test_scores_follow_the_rrf_formula():
    fused = dict(rrf_fuse([["a", "b"]], k=60))
    assert fused["a"] == 1 / 61  # rank 1
    assert fused["b"] == 1 / 62  # rank 2


def test_consensus_beats_a_single_top_hit():
    # 'b' is 2nd in both lists; 'a' is 1st in only one. With k=60 the flat curve
    # rewards appearing in both lists, so consensus 'b' should outrank 'a'.
    fused = rrf_fuse([["a", "b"], ["c", "b"]], k=60)
    assert fused[0][0] == "b"


def test_k_changes_the_consensus_vs_top_hit_tradeoff():
    # 'top' is rank 1 in one list only; 'both' is a DEEP rank 10 in both lists.
    # This is where k actually bites (a shallow consensus item always wins):
    fillers1 = [f"f{i}" for i in range(8)]
    fillers2 = [f"g{i}" for i in range(8)]
    lists = [["top", *fillers1, "both"], ["c", *fillers2, "both"]]
    # 'top' is rank 1 in one list; 'both' is a deep rank 10 in both lists.
    big = dict(rrf_fuse(lists, k=60))
    small = dict(rrf_fuse(lists, k=1))
    assert big["both"] > big["top"]    # large k flattens -> deep consensus wins
    assert small["top"] > small["both"]  # small k -> the single top hit wins


def test_missing_from_a_list_just_contributes_nothing():
    # 'x' only appears in the first list; it still scores, just from one list.
    fused = dict(rrf_fuse([["x", "y"], ["y"]], k=60))
    assert fused["y"] > fused["x"]  # y: two lists, x: one


def test_ties_break_deterministically_by_id():
    # Symmetric input -> equal scores -> stable, id-sorted order (reproducible eval).
    fused = rrf_fuse([["b", "a"], ["a", "b"]], k=60)
    ids = [i for i, _ in fused]
    assert ids == sorted(ids)


def test_empty_input_yields_empty_output():
    assert rrf_fuse([], k=60) == []
    assert rrf_fuse([[], []], k=60) == []
