"""Layer 7 tests: Langfuse instrumentation (D-045).

Offline and hermetic — no real Langfuse account. Two things are checked:
* the **disabled** path (no keys) is a true no-op: null trace with no callbacks, null
  generation, and the gateway still answers + records metrics through it;
* the **enabled** path, with a *fake* client injected, actually opens a root span,
  exposes a callback, records the generation's model + token usage, and flushes once.
"""

from __future__ import annotations

from types import SimpleNamespace

import src.obs.langfuse as lf
from src.llm.gateway import LLMGateway
from src.llm.metrics import collect_metrics
from src.obs import observe_generation, trace_agent

# --- disabled path (autouse conftest patches _client -> None) ---


def test_langfuse_enabled_reflects_keys(monkeypatch):
    monkeypatch.setattr(lf, "get_settings", lambda: SimpleNamespace(
        langfuse_public_key="", langfuse_secret_key=""))
    assert lf.langfuse_enabled() is False
    monkeypatch.setattr(lf, "get_settings", lambda: SimpleNamespace(
        langfuse_public_key="pk", langfuse_secret_key="sk"))
    assert lf.langfuse_enabled() is True


def test_trace_agent_is_noop_when_disabled():
    with trace_agent("q") as tr:
        assert tr.callbacks == []      # nothing to pass into the graph
        tr.set_output({"answer": "a"})  # must not raise


def test_observe_generation_is_noop_when_disabled():
    with observe_generation("groq/llama-3.1-8b-instant") as gen:
        gen.set_result("hello", {"prompt_tokens": 1, "completion_tokens": 2})  # must not raise


def test_gateway_answers_and_records_metrics_with_tracing_disabled(tmp_path, monkeypatch):
    gw = LLMGateway("groq/llama-3.1-8b-instant", cache_dir=str(tmp_path))
    monkeypatch.setattr(
        gw, "_call_once", lambda p: ("ok", {"prompt_tokens": 3, "completion_tokens": 4})
    )
    with collect_metrics() as m:
        out = gw.complete("sys", "obs-disabled-prompt")
    assert out == "ok"
    assert m.calls == 1 and m.total_tokens == 7


# --- enabled path (inject a fake client) ---


class _FakeSpan:
    def __init__(self, rec):
        self.rec = rec

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, **kw):
        self.rec["updates"].append(kw)

    def set_trace_io(self, **kw):
        self.rec["io"].append(kw)


class _FakeClient:
    def __init__(self):
        self.rec = {"obs": [], "updates": [], "io": [], "flushed": 0}

    def start_as_current_observation(self, **kw):
        self.rec["obs"].append(kw)
        return _FakeSpan(self.rec)

    def flush(self):
        self.rec["flushed"] += 1


def _use_fake_client(monkeypatch) -> _FakeClient:
    fake = _FakeClient()
    monkeypatch.setattr(lf, "_client", lambda: fake)
    return fake


def test_observe_generation_records_model_and_usage(monkeypatch):
    fake = _use_fake_client(monkeypatch)
    with observe_generation("gemini/gemini-2.5-flash", input={"user": "hi"}) as gen:
        gen.set_result("answer text", {"prompt_tokens": 11, "completion_tokens": 5})
    # opened as a generation with the model recorded
    obs = fake.rec["obs"][0]
    assert obs["as_type"] == "generation" and obs["model"] == "gemini/gemini-2.5-flash"
    # output + token usage recorded, mapped to Langfuse's input/output usage keys
    upd = fake.rec["updates"][0]
    assert upd["output"] == "answer text"
    assert upd["usage_details"] == {"input": 11, "output": 5}


def test_observe_generation_drops_missing_token_counts(monkeypatch):
    fake = _use_fake_client(monkeypatch)
    with observe_generation("groq/llama-3.1-8b-instant") as gen:
        gen.set_result("x", {"prompt_tokens": 9, "completion_tokens": None})
    assert fake.rec["updates"][0]["usage_details"] == {"input": 9}


def test_trace_agent_opens_span_exposes_callback_and_flushes(monkeypatch):
    fake = _use_fake_client(monkeypatch)

    class _DummyHandler:
        def __init__(self, **kw):
            pass

    monkeypatch.setattr("langfuse.langchain.CallbackHandler", _DummyHandler)
    monkeypatch.setattr(lf, "get_settings", lambda: SimpleNamespace(
        langfuse_public_key="pk", langfuse_secret_key="sk"))

    with trace_agent("what is X?", name="ask") as tr:
        assert len(tr.callbacks) == 1  # a handler to hand to the graph
        tr.set_output({"answer": "done"}, metadata={"metrics": {"llm_calls": 1}})

    obs = fake.rec["obs"][0]
    # root observation opened as an agent with the question as input (trace I/O derives from it)
    assert obs["as_type"] == "agent" and obs["input"] == {"question": "what is X?"}
    assert fake.rec["updates"][-1]["output"] == {"answer": "done"}              # answer recorded
    assert fake.rec["flushed"] == 1                                            # flushed on exit
