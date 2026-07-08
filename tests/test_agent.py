"""Graph-wiring tests for the Layer 5a cited-answer agent.

Offline: retriever + gateway are faked, so the retrieve→synthesize→cite wiring and
the citation attachment are exercised without a live cluster or LLM call (the live
path is the layer's gate). Confirms the graph threads state and resolves citations.
"""

from src.agent.graph import answer_question
from src.agent.nodes import format_context, route_after_verify
from src.retrieval.search import Hit


def _chunks():
    return [
        Hit(id="a", score=0.0, text="Alpha text", source_url="url-a", heading_path="Alpha"),
        Hit(id="b", score=0.0, text="Beta text", source_url="url-b", heading_path="Beta"),
    ]


class _FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, question, top_k=None):
        return self._chunks[: top_k or len(self._chunks)], 1.0


class _FakeGateway:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def complete(self, system, user, *, max_tokens=1024, response_json=False):
        self.calls += 1
        self.last_user = user
        return self.answer


class _SeqGateway:
    """Returns queued replies in order (last reply repeats) — for multi-call retry tests."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0
        self.users: list[str] = []

    def complete(self, system, user, *, max_tokens=1024, response_json=False):
        self.users.append(user)
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return reply


def _grounded():
    return _FakeGateway('{"grounded": true}')


def test_graph_retrieves_synthesizes_and_resolves_citations():
    state = answer_question(
        "how?",
        retriever=_FakeRetriever(_chunks()),
        gateway=_FakeGateway("Claim one [1]. Claim two [2]."),
        verifier=_grounded(),
    )
    assert state["answer"] == "Claim one [1]. Claim two [2]."
    assert [c.chunk_id for c in state["citations"]] == ["a", "b"]
    assert state["invalid_citations"] == []


def test_graph_flags_hallucinated_citation():
    state = answer_question(
        "how?",
        retriever=_FakeRetriever(_chunks()),
        gateway=_FakeGateway("Real [1] but invented [9]."),
        verifier=_grounded(),
    )
    assert [c.marker for c in state["citations"]] == [1]
    assert state["invalid_citations"] == [9]


def test_graph_refuses_when_no_context():
    state = answer_question(
        "how?",
        retriever=_FakeRetriever([]),
        gateway=_FakeGateway("should not be used"),
        verifier=_grounded(),
    )
    assert state["answer"] == "I don't know based on the provided documentation."
    assert state["citations"] == []


def test_verify_refuses_ungrounded_answer_and_drops_citations():
    # synthesis drafts a cited answer, but the verifier says it isn't grounded ->
    # with no retry budget (max_retries=0 == 5b behavior) the agent refuses and no
    # citations survive (the D-036 "Paris [1,2]" fix).
    state = answer_question(
        "capital of France?",
        retriever=_FakeRetriever(_chunks()),
        gateway=_FakeGateway("The capital is Paris [1, 2]."),
        verifier=_FakeGateway('{"grounded": false}'),
        max_retries=0,
    )
    assert state["grounded"] is False
    assert state["answer"] == "I don't know based on the provided documentation."
    assert state["citations"] == []


def test_verify_keeps_grounded_answer():
    state = answer_question(
        "how?",
        retriever=_FakeRetriever(_chunks()),
        gateway=_FakeGateway("Grounded claim [1]."),
        verifier=_FakeGateway('{"grounded": true}'),
    )
    assert state["grounded"] is True
    assert state["answer"] == "Grounded claim [1]."
    assert [c.chunk_id for c in state["citations"]] == ["a"]


def test_format_context_numbers_passages():
    ctx = format_context(_chunks())
    assert "[1] Alpha" in ctx and "[2] Beta" in ctx
    assert "url-a" in ctx and "url-b" in ctx


# --- Layer 5c: self-correction control router + retry loop (D-041) ---


def test_router_grounded_goes_to_cite():
    assert route_after_verify({"grounded": True, "retries": 0, "max_retries": 1}) == "cite"


def test_router_ungrounded_retries_while_budget_remains():
    assert route_after_verify({"grounded": False, "retries": 0, "max_retries": 1}) == "retry"


def test_router_ungrounded_refuses_when_budget_spent():
    assert route_after_verify({"grounded": False, "retries": 1, "max_retries": 1}) == "refuse"


def test_router_zero_budget_refuses_immediately():
    # max_retries=0 reduces the loop to 5b: ungrounded -> refuse, no retry.
    assert route_after_verify({"grounded": False, "retries": 0, "max_retries": 0}) == "refuse"


def test_self_correction_rescues_a_fixable_draft():
    # First draft over-claims (verifier: not grounded); after one feedback-guided retry the
    # model produces a grounded answer -> the agent keeps it instead of refusing (D-041).
    synth = _SeqGateway(["Over-claim Paris [1].", "Grounded fix [1]."])
    verifier = _SeqGateway(['{"grounded": false}', '{"grounded": true}'])
    state = answer_question(
        "how?",
        retriever=_FakeRetriever(_chunks()),
        gateway=synth,
        verifier=verifier,
        max_retries=1,
    )
    assert state["answer"] == "Grounded fix [1]."
    assert state["grounded"] is True
    assert state["retries"] == 1
    assert [c.chunk_id for c in state["citations"]] == ["a"]
    # the retry prompt carried corrective feedback the first one lacked
    assert "NOT supported" in synth.users[1]
    assert "NOT supported" not in synth.users[0]


def test_self_correction_refuses_after_exhausting_budget():
    # An unfixable draft (verifier always false) is retried up to the budget, then refused.
    synth = _SeqGateway(["Over-claim Paris [1]."])
    verifier = _FakeGateway('{"grounded": false}')
    state = answer_question(
        "capital of France?",
        retriever=_FakeRetriever(_chunks()),
        gateway=synth,
        verifier=verifier,
        max_retries=2,
    )
    assert state["answer"] == "I don't know based on the provided documentation."
    assert state["grounded"] is False
    assert state["retries"] == 2
    assert state["citations"] == []
    assert synth.calls == 3  # initial draft + 2 retries
