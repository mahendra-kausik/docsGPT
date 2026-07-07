"""Unit tests for the Layer 3 eval harness (metrics, gold I/O, compile).

Pure/offline: no torch, no live Qdrant, no LLM — so `./tasks.ps1 test` stays fast
and the retrieval metrics (the project's most defensible numbers) are pinned to
known values. The live dense-search path is exercised by the Layer 3 eval run.
"""

import math

from src.eval import metrics as M
from src.eval.build_gold import build
from src.eval.compile_gold import compile_gold, parse_decisions, resolve_gold_ids
from src.eval.gold import UNANSWERABLE, VERIFIED, ForumSeed, GoldItem, html_to_text, save_gold
from src.eval.prefill import answer_link_matches, apply_prefill, normalize_docs_url
from src.eval.propose import Candidate, Proposal
from src.eval.synth import _clean_question
from src.llm.gateway import _split_provider

RANKED = ["a", "b", "c", "d", "e"]  # retriever output, best-first


def test_hit_and_recall_single_gold():
    assert M.hit_at_k(RANKED, ["c"], 5) == 1.0
    assert M.hit_at_k(RANKED, ["c"], 2) == 0.0  # c is at rank 3, outside top-2
    assert M.recall_at_k(RANKED, ["c"], 5) == 1.0  # single gold found
    assert M.recall_at_k(RANKED, ["z"], 5) == 0.0  # gold absent


def test_recall_multi_gold_is_fraction():
    # two gold chunks, only one inside the top-3
    assert M.recall_at_k(RANKED, ["a", "e"], 3) == 0.5
    assert M.recall_at_k(RANKED, ["a", "e"], 5) == 1.0


def test_reciprocal_rank_uses_first_hit_within_cutoff():
    assert M.reciprocal_rank(RANKED, ["b"], 3) == 0.5  # first hit at rank 2
    assert M.reciprocal_rank(RANKED, ["a"], 3) == 1.0  # rank 1
    assert M.reciprocal_rank(RANKED, ["d"], 3) == 0.0  # rank 4, past the mrr cutoff


def test_ndcg_perfect_and_discounted():
    # gold at rank 1 -> perfect nDCG
    assert M.ndcg_at_k(RANKED, ["a"], 5) == 1.0
    # gold at rank 2 -> DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    expected = (1.0 / math.log2(3)) / 1.0
    assert math.isclose(M.ndcg_at_k(RANKED, ["b"], 5), expected)
    assert M.ndcg_at_k(RANKED, ["z"], 5) == 0.0


def test_score_query_keys_and_aggregate():
    q = M.score_query(RANKED, ["a"])
    assert q["recall@1"] == 1.0 and q["mrr@3"] == 1.0
    q2 = M.score_query(RANKED, ["z"])  # all-miss
    agg = M.aggregate([q, q2])
    assert agg["recall@1"] == 0.5  # macro-average of 1.0 and 0.0
    assert M.aggregate([]) == {}


def test_gold_item_is_scored_gate():
    assert GoldItem(qid=1, question="q", gold_chunk_ids=["a"], status=VERIFIED).is_scored
    assert not GoldItem(qid=2, question="q", gold_chunk_ids=[], status=VERIFIED).is_scored
    assert not GoldItem(qid=3, question="q", gold_chunk_ids=[], status=UNANSWERABLE).is_scored


def test_html_to_text_flattens_blocks_and_entities():
    cooked = "<p>Use <code>ainvoke</code> &amp; stream.</p><pre>x = 1</pre>"
    text = html_to_text(cooked)
    assert "ainvoke" in text and "&amp;" not in text and "&" in text
    assert "<" not in text


def _proposal(qid=10):
    return Proposal(
        qid=qid,
        question="how?",
        source_url="u",
        candidates=[
            Candidate(
                rank=1,
                chunk_id="aaaa1111",
                score=0.9,
                heading_path="H",
                source_url="u",
                snippet="s",
            ),
            Candidate(
                rank=2,
                chunk_id="bbbb2222",
                score=0.8,
                heading_path="H",
                source_url="u",
                snippet="s",
            ),
        ],
    )


def test_resolve_gold_ids_ranks_hex_and_dedup():
    prop = _proposal()
    assert resolve_gold_ids("1 2", prop) == ["aaaa1111", "bbbb2222"]
    assert resolve_gold_ids("2 2", prop) == ["bbbb2222"]  # dedup
    assert resolve_gold_ids("cccc3333", prop) == ["cccc3333"]  # literal id not in pool


def test_resolve_gold_ids_rejects_bad_tokens():
    prop = _proposal()
    for bad in ("9", "nope"):  # rank out of range / non-hex word
        try:
            resolve_gold_ids(bad, prop)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_parse_decisions_and_compile():
    review = (
        "## [10] title\n"
        "> DECISION [qid=10]: 1\n"
        "## [11] title\n"
        "> DECISION [qid=11]: x\n"
        "## [12] title\n"
        "> DECISION [qid=12]: \n"  # pending
    )
    decisions = parse_decisions(review)
    assert decisions == {10: "1", 11: "x", 12: ""}

    proposals = {10: _proposal(10), 11: _proposal(11), 12: _proposal(12)}
    items, counts = compile_gold(proposals, decisions)
    assert counts == {"verified": 1, "unanswerable": 1, "pending": 1}
    scored = [it for it in items if it.is_scored]
    assert len(scored) == 1 and scored[0].qid == 10 and scored[0].gold_chunk_ids == ["aaaa1111"]


def test_normalize_docs_url_strips_lang_segment_and_anchor():
    # rendered answer link vs our chunk url differ only by the /python/ segment
    a = normalize_docs_url("https://docs.langchain.com/oss/python/langgraph/persistence#foo")
    b = normalize_docs_url("https://docs.langchain.com/oss/langgraph/persistence/")
    assert a == b == "docs.langchain.com/oss/langgraph/persistence"
    # a langsmith link (outside the oss corpus) is left as-is -> won't match oss chunks
    assert "langsmith" in normalize_docs_url("https://docs.langchain.com/langsmith/x")


def test_answer_link_matches_one_best_chunk_per_linked_page(monkeypatch):
    import src.eval.prefill as P

    # answer links one python docs page; the pool has 2 chunks of that page + 1 other
    base = "https://docs.langchain.com/oss"
    monkeypatch.setattr(
        P,
        "accepted_answer_html",
        lambda seed: f'<a href="{base}/python/langgraph/persistence">docs</a>',
    )

    def _c(rank, cid, path):
        return Candidate(
            rank=rank,
            chunk_id=cid,
            score=1.0 / rank,
            heading_path="H",
            source_url=f"{base}/{path}",
            snippet="s",
        )

    prop = Proposal(
        qid=10,
        question="how?",
        source_url="u",
        candidates=[
            _c(1, "a", "langchain/models"),  # unrelated page
            _c(2, "b", "langgraph/persistence"),  # best chunk of the linked page
            _c(5, "c", "langgraph/persistence"),  # another chunk of the same page
        ],
    )
    # only the best chunk (rank 2) of the linked page — not rank 5 too, not the unrelated rank 1
    assert answer_link_matches(ForumSeed(id=10, title="t", url="u"), prop) == [2]


def test_apply_prefill_fills_blank_and_is_idempotent():
    md = (
        "> DECISION [qid=10]: \n"
        "> DECISION [qid=11]: 4\n"  # already decided -> must be left alone
    )
    out, filled = apply_prefill(md, {10: [2], 11: [1]})
    assert filled == 1
    assert "> DECISION [qid=10]: 2" in out
    assert "auto-suggested" in out  # annotation inserted above the filled line
    assert "> DECISION [qid=11]: 4" in out  # human edit preserved
    # re-running does not double-fill or duplicate the annotation
    out2, filled2 = apply_prefill(out, {10: [2], 11: [1]})
    assert filled2 == 0 and out2.count("auto-suggested") == 1


# --- D-025: synthetic generation + merge ---


def test_split_provider():
    assert _split_provider("groq/llama-3.1-8b-instant") == ("groq", "llama-3.1-8b-instant")
    assert _split_provider("gemini/gemini-2.5-flash") == ("gemini", "gemini-2.5-flash")
    assert _split_provider("llama-3.1-8b-instant") == ("groq", "llama-3.1-8b-instant")


def test_clean_question_accepts_good_and_rejects_bad():
    assert _clean_question('{"question": "How do I stream tokens from a chat model?"}') == (
        "How do I stream tokens from a chat model?"
    )
    # rejects: not a question, too short, meta-leak, malformed JSON
    assert _clean_question('{"question": "This is a statement."}') is None
    assert _clean_question('{"question": "Why?"}') is None
    assert _clean_question('{"question": "What does the passage say about tools?"}') is None
    assert _clean_question("not json") is None


def test_build_merges_scored_items_by_source(tmp_path):
    forum = [GoldItem(qid=1, question="q", gold_chunk_ids=["a"], source="forum")]
    synth = [
        GoldItem(
            qid=1_000_000, question="q", gold_chunk_ids=["b"], source="synthetic", hop="single"
        ),
        # empty-gold synthetic item is not scored -> excluded from the merge:
        GoldItem(qid=1_000_001, question="q", gold_chunk_ids=[], source="synthetic", hop="multi"),
    ]
    fp, sp = tmp_path / "forum.jsonl", tmp_path / "synth.jsonl"
    save_gold(forum, str(fp))
    save_gold(synth, str(sp))
    items, report = build(str(fp), str(sp))
    assert report["total"] == 2
    assert report["by_source"] == {"forum": 1, "synthetic": 1}
    assert report["by_hop"] == {"single": 1}
