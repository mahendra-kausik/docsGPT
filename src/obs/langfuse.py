"""Langfuse instrumentation for the agent (Layer 7, D-045).

Two things are traced, because our LLM path deliberately bypasses LangChain's chat
models (the gateway calls the Groq/Gemini SDKs directly, D-033):

* **Node spans** — a Langfuse LangChain ``CallbackHandler`` is passed into the graph
  run, so every LangGraph node (retrieve → synthesize → verify → retry/refuse → cite)
  renders as a nested span. This is the "show the loop" gate.
* **Generation observations** — ``observe_generation()`` wraps each ``LLMGateway``
  network call, recording model + token usage so Langfuse can attribute cost/latency
  per LLM call. The callback handler cannot see these (they are not LangChain LLMs),
  so we emit them by hand; opened inside a node's execution, they nest under it.

Everything is a **no-op when the Langfuse keys are blank** (`langfuse_enabled()` is
False): `trace_agent()` yields a null trace with no callbacks and `observe_generation()`
yields a null generation, so tests, the CLI, and the eval harness need no Langfuse
account. Client construction is wrapped defensively — an observability failure must
never take down an answer.

Threading: the root span is opened inside the request thread (the sync `/ask`
threadpool worker, or the `/ask/stream` producer thread), so the OTel context and the
per-request metrics contextvar (D-043) live in the same thread and nest correctly.
Traces are **flushed per request** because Cloud Run scales to zero (Layer 8) — the
batched exporter must ship before the instance may be frozen/killed.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)


def langfuse_enabled() -> bool:
    """True only when both Langfuse keys are configured (secrets present in .env)."""
    s = get_settings()
    return bool(s.langfuse_public_key and s.langfuse_secret_key)


@lru_cache
def _client():
    """Return a cached Langfuse client, or None if disabled/misconfigured.

    Cached so one client (one background exporter) is shared process-wide. Any
    construction error disables tracing rather than propagating — observability is
    best-effort and must never break the request path.
    """
    if not langfuse_enabled():
        return None
    try:
        from langfuse import Langfuse

        s = get_settings()
        return Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
            tracing_enabled=True,
        )
    except Exception as exc:  # noqa: BLE001 — never let tracing setup break answering
        logger.warning("Langfuse disabled: client init failed (%s)", exc)
        return None


class _NullTrace:
    """No-op trace used when Langfuse is disabled: no callbacks, output ignored."""

    callbacks: list = []

    def set_output(self, output: Any, **_: Any) -> None:  # noqa: D401
        pass


class _Trace:
    """Live trace: exposes the LangChain callback and records the trace-level output."""

    def __init__(self, span: Any, callbacks: list) -> None:
        self._span = span
        self.callbacks = callbacks

    def set_output(self, output: Any, *, metadata: dict | None = None) -> None:
        """Attach the final answer/metrics to the root span (Langfuse derives trace I/O)."""
        try:
            self._span.update(output=output, metadata=metadata)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Langfuse set_output failed: %s", exc)


@contextmanager
def trace_agent(question: str, *, name: str = "agent", metadata: dict | None = None):
    """Scope a root Langfuse trace over one agent run; yields a trace with `.callbacks`.

    Pass ``trace.callbacks`` into the graph config so every node becomes a span. When
    Langfuse is disabled this yields a null trace (empty callbacks), so the caller's
    code path is identical with or without observability. Flushes on exit.
    """
    client = _client()
    cm = handler = None
    if client is not None:
        # Build the handler + open the span up front; any setup failure falls back to the
        # null path so a tracing glitch never breaks the request (yields exactly once).
        try:
            from langfuse.langchain import CallbackHandler

            handler = CallbackHandler(public_key=get_settings().langfuse_public_key)
            cm = client.start_as_current_observation(
                name=name, as_type="agent", input={"question": question}, metadata=metadata
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse trace disabled for this run: %s", exc)
            cm = None
    if cm is None:
        yield _NullTrace()
        return
    with cm as span:
        try:
            yield _Trace(span, [handler])
        finally:
            try:
                client.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Langfuse flush failed: %s", exc)


class _NullGen:
    """No-op generation used when Langfuse is disabled."""

    def set_result(self, output: str, usage: dict | None = None) -> None:
        pass


class _Gen:
    """Live generation observation: records model output + token usage on exit."""

    def __init__(self, gen: Any) -> None:
        self._gen = gen

    def set_result(self, output: str, usage: dict | None = None) -> None:
        """Record the completion text and (best-effort) prompt/completion token counts."""
        usage_details = None
        if usage:
            pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
            usage_details = {
                k: v
                for k, v in {"input": pt, "output": ct}.items()
                if v is not None
            } or None
        try:
            self._gen.update(output=output, usage_details=usage_details)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Langfuse generation update failed: %s", exc)


@contextmanager
def observe_generation(model: str, *, input: Any = None, model_parameters: dict | None = None):
    """Wrap one LLM gateway call as a Langfuse generation (no-op when disabled).

    Opened inside a node's execution, so it nests under that node's span. Cost is
    attributed by Langfuse from ``model`` + ``usage_details`` (notional on free tier);
    tokens and latency — the hard engineering metrics (PLAN §6) — are always recorded.
    """
    client = _client()
    cm = None
    if client is not None:
        try:
            cm = client.start_as_current_observation(
                name="llm",
                as_type="generation",
                model=model,
                input=input,
                model_parameters=model_parameters,
            )
        except Exception as exc:  # noqa: BLE001 — tracing setup must never break the call
            logger.debug("Langfuse generation span failed: %s", exc)
            cm = None
    if cm is None:
        yield _NullGen()
        return
    with cm as gen:
        yield _Gen(gen)
