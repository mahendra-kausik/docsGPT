"""Layer 6 tests: metrics accounting, 429 backoff recovery, and the API surface (D-043).

Offline: the agent (`answer_question`/`stream_events`) is faked at the app boundary so the
FastAPI wiring, SSE framing, and metrics rendering are exercised without a live cluster or LLM
call. The gateway's backoff is tested against a *simulated* 429 (the layer's gate) with sleep
patched out. The live cited+streamed answer is the manual gate, not a unit test.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import RateLimitError

import src.api.app as app_module
from src.api.app import app
from src.llm.gateway import LLMGateway
from src.llm.metrics import collect_metrics, record_cache_hit, record_call

client = TestClient(app)


# --- metrics accumulator (D-043) ---


def test_metrics_totals_calls_tokens_and_cache_hits():
    with collect_metrics() as m:
        record_call(10, 5)
        record_call(20, 8)
        record_cache_hit()
    assert m.calls == 2
    assert m.cache_hits == 1
    assert m.total_tokens == 43
    assert m.as_dict()["total_tokens"] == 43


def test_metrics_recording_is_noop_outside_scope():
    # No active scope -> recording must not raise (CLI / eval paths call the gateway too).
    record_call(1, 1)
    record_cache_hit()


# --- gateway backoff recovers from a simulated 429 (the gate) ---


def test_backoff_recovers_from_simulated_429(tmp_path, monkeypatch):
    gw = LLMGateway("groq/llama-3.1-8b-instant", cache_dir=str(tmp_path))
    monkeypatch.setattr("src.llm.gateway.time.sleep", lambda *_: None)  # no real delay in tests

    resp = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
    attempts = {"n": 0}

    def flaky_call_once(payload):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RateLimitError("rate limited", response=resp, body=None)
        return "recovered", {"prompt_tokens": 3, "completion_tokens": 4}

    monkeypatch.setattr(gw, "_call_once", flaky_call_once)

    with collect_metrics() as m:
        out = gw.complete("sys", "unique-user-prompt-xyz")

    assert out == "recovered"
    assert attempts["n"] == 2          # first 429 was retried, second call succeeded
    assert m.calls == 1                # one *successful* real call recorded
    assert m.total_tokens == 7


def test_backoff_reraises_non_retryable(tmp_path, monkeypatch):
    gw = LLMGateway("groq/llama-3.1-8b-instant", cache_dir=str(tmp_path))

    def bad_request(payload):
        raise ValueError("400 bad request")  # not in the retryable set

    monkeypatch.setattr(gw, "_call_once", bad_request)
    with pytest.raises(ValueError):
        gw.complete("sys", "another-unique-prompt")


# --- API endpoints (agent faked at the app boundary) ---


def _fake_state():
    from src.agent.citations import Citation

    return {
        "answer": "You can stream tokens [1].",
        "grounded": True,
        "retries": 0,
        "citations": [Citation(1, "chunk-a", "https://docs/x", "Streaming")],
        "invalid_citations": [],
    }


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_ask_returns_answer_citations_and_metrics(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "answer_question",
        lambda q, gateway=None, max_retries=None, config=None: _fake_state(),
    )
    r = client.post("/ask", json={"question": "how do I stream tokens?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "You can stream tokens [1]."
    assert body["grounded"] is True
    assert body["citations"][0]["chunk_id"] == "chunk-a"
    assert body["citations"][0]["source_url"] == "https://docs/x"
    # metrics envelope is always present (zero here — the agent is faked, no gateway calls)
    for key in ("llm_calls", "cache_hits", "total_tokens", "latency_ms"):
        assert key in body["metrics"]


def test_ask_synthesis_model_selects_gateway(monkeypatch):
    # A selected synthesis_model reaches the graph as a gateway on that provider/model (D-046).
    seen = {}

    def fake(q, gateway=None, max_retries=None, config=None):
        seen["gateway"] = gateway
        return _fake_state()

    monkeypatch.setattr(app_module, "answer_question", fake)
    r = client.post(
        "/ask", json={"question": "x", "synthesis_model": "groq/llama-3.3-70b-versatile"}
    )
    assert r.status_code == 200
    gw = seen["gateway"]
    assert gw is not None and gw.provider == "groq" and gw.model == "llama-3.3-70b-versatile"


def test_ask_default_lets_graph_pick_synthesizer(monkeypatch):
    # No synthesis_model -> gateway is None so the graph builds the config default (Groq 70B).
    seen = {}

    def fake(q, gateway=None, max_retries=None, config=None):
        seen["gateway"] = gateway
        return _fake_state()

    monkeypatch.setattr(app_module, "answer_question", fake)
    client.post("/ask", json={"question": "x"})
    assert seen["gateway"] is None


def test_ask_stream_emits_stages_tokens_and_done(monkeypatch):
    from src.agent.citations import Citation
    from src.retrieval.search import Hit

    def fake_stream_events(question, *, gateway=None, max_retries=None, config=None):
        hit = Hit(id="a", score=0.0, text="t", source_url="u", heading_path="H")
        yield "retrieve", {"chunks": [hit]}
        yield "synthesize", {"answer": "You can stream tokens [1]."}
        yield "verify", {"grounded": True}
        yield "cite", {
            "citations": [Citation(1, "chunk-a", "https://docs/x", "Streaming")],
            "invalid_citations": [],
        }

    monkeypatch.setattr(app_module, "stream_events", fake_stream_events)
    r = client.post("/ask/stream", json={"question": "how do I stream tokens?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    text = r.text

    # lifecycle events in order
    assert "event: stage" in text
    assert '"stage": "retrieve"' in text and '"chunks": 1' in text
    assert '"stage": "verify"' in text and '"grounded": true' in text
    # the verified answer arrived as token frames that reassemble to the full answer
    assert "event: token" in text
    assert "event: done" in text
    # a done payload carries the resolved citation + metrics envelope
    assert '"chunk_id": "chunk-a"' in text
    assert '"latency_ms"' in text


def test_ask_stream_reports_error_frame(monkeypatch):
    def boom(question, *, gateway=None, max_retries=None, config=None):
        raise RuntimeError("kaboom")
        yield  # pragma: no cover — make it a generator

    monkeypatch.setattr(app_module, "stream_events", boom)
    r = client.post("/ask/stream", json={"question": "x"})
    assert r.status_code == 200
    assert "event: error" in r.text
    assert "kaboom" in r.text
