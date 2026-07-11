"""FastAPI surface for the cited-answer agent: JSON + SSE streaming (Layer 6, D-043).

Two endpoints wrap the same LangGraph agent:

* ``POST /ask``          — run to completion, return the verified answer + citations + metrics.
* ``POST /ask/stream``   — Server-Sent Events: ``stage`` events as each node completes, then
  the *verified* answer streamed word-by-word as ``token`` events, then a ``done`` event with
  citations + per-request metrics.

We stream **lifecycle events and the verified answer**, not raw synthesis tokens: grounding
verification is post-synthesis (5b/5c) and may retract or refuse a draft, so streaming the raw
draft would show the user a hallucination we then take back (D-043). Every LLM call is scoped by
`collect_metrics()` so the response reports LLM calls/query, tokens/query, and end-to-end latency.

The streaming endpoint runs the blocking graph in a worker thread and hands events to the event
loop over an `asyncio.Queue`, so the loop is never blocked and the metrics contextvar stays intact
for the whole run (it is set inside the one thread that makes every gateway call).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from collections.abc import Iterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent.graph import answer_question, stream_events
from src.config import get_settings
from src.llm.metrics import collect_metrics
from src.obs import trace_agent

app = FastAPI(title="DocsGPT-Agent", version="0.6.0")

# The UI is served from a separate Vercel origin, so the browser needs CORS (Layer 9, D-049).
# CORS_ORIGINS is a comma-separated allowlist set on Cloud Run; falls back to the Vite dev
# origin so local `npm run dev` works with no config.
_ALLOWED_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "").split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS or ["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Split into words keeping trailing whitespace, so the concatenated token stream reproduces
# the answer exactly (no lost/added spaces).
_WORD = re.compile(r"\S+\s*")


class AskRequest(BaseModel):
    """Body for both ask endpoints."""

    question: str
    max_retries: int | None = None  # None -> config default (agent_max_retries, D-041)
    # None -> config default synthesizer (Groq 70B, D-046); provider-prefixed to opt into
    # Gemini per request, e.g. "gemini/gemini-2.5-flash" (burns its 20/day free cap — demo only).
    synthesis_model: str | None = None


def _synthesis_gateway(model: str | None):
    """Build a per-request synthesis gateway when the client selected a model (Layer 8a, D-046).

    None -> the graph builds the deployed default (config synthesis_model) itself.
    """
    if not model:
        return None
    from src.llm.gateway import LLMGateway

    return LLMGateway(model)


def _citations_payload(state: dict) -> list[dict]:
    """JSON-safe view of the resolved citations on the final state."""
    return [
        {
            "marker": c.marker,
            "chunk_id": c.chunk_id,
            "heading_path": c.heading_path,
            "source_url": c.source_url,
        }
        for c in state.get("citations", [])
    ]


def _answer_payload(state: dict, metrics: dict) -> dict:
    """Assemble the non-streamed response body (also the SSE ``done`` payload)."""
    return {
        "answer": state.get("answer", ""),
        "grounded": state.get("grounded", True),
        "retries": state.get("retries", 0),
        "citations": _citations_payload(state),
        "invalid_citations": state.get("invalid_citations", []),
        "metrics": metrics,
    }


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stage_event(node: str, delta: dict) -> dict | None:
    """Map a graph node's completion into a client-facing stage event (or None to skip)."""
    if node == "retrieve":
        return {"stage": "retrieve", "chunks": len(delta.get("chunks", []))}
    if node == "synthesize":
        return {"stage": "synthesize"}
    if node == "verify":
        return {"stage": "verify", "grounded": delta.get("grounded")}
    if node == "retry":
        return {"stage": "retry", "n": delta.get("retries")}
    if node == "refuse":
        return {"stage": "refuse"}
    return None  # 'cite' is folded into the final 'done' event


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe (Cloud Run health check, Layer 8)."""
    return {"status": "ok"}


@app.get("/ping")
def ping() -> dict:
    """Keep-alive that touches Qdrant so its free cluster's idle timer resets (Layer 8c, D-004).

    A Cloud Scheduler job hits this weekly. A cheap `count` is a real Qdrant request (unlike
    /healthz, which never reaches the store) but loads no models, so it stays fast and free.
    """
    from src.retrieval.index import get_client

    s = get_settings()
    count = get_client().count(s.qdrant_hybrid_collection, exact=False).count
    return {"status": "ok", "collection": s.qdrant_hybrid_collection, "points": count}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Answer a question and return the verified answer, citations, and per-request metrics.

    A sync endpoint runs in Starlette's threadpool, so the whole agent run (and every
    gateway call it makes) shares one thread and one metrics scope.
    """
    t0 = time.perf_counter()
    with trace_agent(req.question, name="ask") as tr:
        with collect_metrics() as m:
            state = answer_question(
                req.question,
                gateway=_synthesis_gateway(req.synthesis_model),
                max_retries=req.max_retries,
                config={"callbacks": tr.callbacks},
            )
        metrics = {**m.as_dict(), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        payload = _answer_payload(state, metrics)
        tr.set_output(payload, metadata={"metrics": metrics})
    return payload


def _iter_words(text: str) -> Iterator[str]:
    """Yield the answer as word-with-trailing-space tokens for the SSE token stream."""
    return iter(_WORD.findall(text))


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """Stream stage events, then the verified answer word-by-word, then citations + metrics."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    def produce() -> None:
        """Run the blocking graph in a worker thread, pushing SSE frames onto the queue."""
        put = lambda item: loop.call_soon_threadsafe(queue.put_nowait, item)  # noqa: E731
        try:
            t0 = time.perf_counter()
            state: dict = {}
            # Root trace opened in THIS worker thread so the Langfuse OTel context and the
            # metrics contextvar share the one thread every gateway call runs on (D-043/D-045).
            with trace_agent(req.question, name="ask/stream") as tr:
                with collect_metrics() as m:
                    for node, delta in stream_events(
                        req.question,
                        gateway=_synthesis_gateway(req.synthesis_model),
                        max_retries=req.max_retries,
                        config={"callbacks": tr.callbacks},
                    ):
                        state.update(delta)
                        ev = _stage_event(node, delta)
                        if ev is not None:
                            put(("stage", ev))
                    metrics = {
                        **m.as_dict(),
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    }
                payload = _answer_payload(state, metrics)
                tr.set_output(payload, metadata={"metrics": metrics})
            for word in _iter_words(state.get("answer", "")):
                put(("token", {"text": word}))
            put(("done", payload))
        except Exception as exc:  # noqa: BLE001 — surface any failure to the client, then close
            put(("error", {"detail": f"{type(exc).__name__}: {exc}"}))
        finally:
            put(sentinel)

    async def event_stream() -> Iterator[str]:
        threading.Thread(target=produce, name="ask-stream", daemon=True).start()
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            event, data = item
            yield _sse(event, data)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
