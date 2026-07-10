"""Observability (Layer 7): Langfuse tracing for the agent graph + LLM gateway.

Public surface kept tiny and import-cheap so the rest of the codebase depends on
these three names, not on Langfuse internals — and so everything is a safe no-op
when Langfuse keys are absent (tests, CLI `ask`, and the eval harness run unchanged).
"""

from src.obs.langfuse import langfuse_enabled, observe_generation, trace_agent

__all__ = ["langfuse_enabled", "observe_generation", "trace_agent"]
