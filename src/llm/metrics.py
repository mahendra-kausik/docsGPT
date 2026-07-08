"""Per-request LLM accounting: calls, tokens, cache hits (Layer 6, D-043).

PLAN §6 wants LLM-calls/query and tokens/query measured. The gateway is the single
choke point every call flows through, so it records here — but the *scope* of a
"query" is set by the caller (one API request may fan out to several nodes). A
`contextvars` collector lets `collect_metrics()` wrap a whole agent run and total
the calls its nodes make, without threading a metrics object through every node.

Cache hits are counted separately and contribute **zero** tokens/calls — a cached
answer spends no quota, and reporting it as such keeps the numbers honest (a run
that is 90% cache hits should not look like it burned 90% of the quota).
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

# None when no collector is active: the gateway still works outside a request scope
# (CLI `ask`, eval harness) — recording is simply a no-op there.
_current: contextvars.ContextVar[LLMMetrics | None] = contextvars.ContextVar(
    "llm_metrics", default=None
)


@dataclass
class LLMMetrics:
    """Running totals for the LLM calls made inside one `collect_metrics()` scope."""

    calls: int = 0            # real provider calls (cache misses that hit the network)
    cache_hits: int = 0       # requests served from the on-disk cache (0 tokens, 0 quota)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int]:
        """JSON-safe view for the API response / bench output."""
        return {
            "llm_calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@contextmanager
def collect_metrics() -> Iterator[LLMMetrics]:
    """Scope a fresh accumulator over a block; nested gateway calls record into it."""
    m = LLMMetrics()
    token = _current.set(m)
    try:
        yield m
    finally:
        _current.reset(token)


def record_call(prompt_tokens: int | None, completion_tokens: int | None) -> None:
    """Count one real provider call and its token usage (no-op outside a scope)."""
    m = _current.get()
    if m is not None:
        m.calls += 1
        m.prompt_tokens += prompt_tokens or 0
        m.completion_tokens += completion_tokens or 0


def record_cache_hit() -> None:
    """Count one cache-served request (spends no quota; no-op outside a scope)."""
    m = _current.get()
    if m is not None:
        m.cache_hits += 1
