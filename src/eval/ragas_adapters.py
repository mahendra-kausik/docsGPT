"""RAGAS adapters: run RAGAS with OUR gateway as judge + OUR bge embeddings (Layer 5d).

Two jobs, both in service of D-040 (RAGAS answer-quality eval, judge fixed to Groq-8B):

1. **Compat shim.** ragas 0.4.3 hard-imports `langchain_community.chat_models.vertexai`
   (and `langchain_community.llms.VertexAI`) at module load — paths removed in the
   langchain-community 0.4.x that our langchain-core 1.x stack pulls. ragas only uses them
   as `isinstance` sentinels (`is_multiple_completion_supported`), so we inject harmless
   stubs BEFORE importing ragas. This keeps our pinned stack (langgraph 1.2.8, groq 1.5.0)
   untouched — no downgrade to the groq 0.37.x that `langchain-groq` would force.

2. **Route the judge through our gateway (D-008/§4).** Instead of `langchain-groq`, we wrap
   `LLMGateway` (fixed `cheap_model` = Groq-8B) as a RAGAS LLM and our bge `Embedder` as the
   RAGAS embeddings. So every RAGAS judge call inherits the gateway's backoff + on-disk cache
   — the rate-limit-safe, reproducible single call path CLAUDE.md §4 mandates — and a second
   Groq account is unnecessary. The gateway cache also makes re-scoring free.
"""

from __future__ import annotations

import sys
import types


def _install_vertexai_shim() -> None:
    """Stub the removed VertexAI symbols ragas 0.4.3 imports (see module docstring)."""
    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})  # sentinel only; never instantiated
        sys.modules[mod_name] = stub
    import langchain_community.llms as community_llms

    if not hasattr(community_llms, "VertexAI"):
        community_llms.VertexAI = type("VertexAI", (), {})


_install_vertexai_shim()

import asyncio  # noqa: E402 — must follow the shim so the ragas import below succeeds

from langchain_core.outputs import Generation, LLMResult  # noqa: E402
from ragas.embeddings.base import BaseRagasEmbeddings  # noqa: E402
from ragas.llms.base import BaseRagasLLM  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.llm.gateway import LLMGateway  # noqa: E402
from src.retrieval.embedder import Embedder  # noqa: E402

# The judge is a plain instruction-follower; RAGAS's own prompts carry the task. Keeping
# this system message neutral lets those prompts drive faithfulness/relevancy scoring.
_JUDGE_SYSTEM = (
    "You are a meticulous evaluation assistant. Follow the instructions in the message "
    "exactly and reply with only what is requested (no preamble)."
)


class GatewayRagasLLM(BaseRagasLLM):
    """A RAGAS LLM backed by our LLMGateway (fixed Groq-8B judge, backoff + cache, D-008)."""

    def __init__(self, gateway: LLMGateway | None = None, max_tokens: int = 1024) -> None:
        super().__init__(cache=None)  # gateway owns caching; skip ragas's own cacher
        # More retries than the default: Groq's 6000 TPM free tier throttles sustained RAGAS
        # traffic with 429s, and the backoff needs to span a full ~60s per-minute window.
        self.gateway = gateway or LLMGateway(get_settings().cheap_model, max_retries=8)
        self.max_tokens = max_tokens

    def is_finished(self, response: LLMResult) -> bool:
        """Gateway returns only completed text (max_tokens is generous) — always finished."""
        return True

    def generate_text(
        self, prompt, n: int = 1, temperature=None, stop=None, callbacks=None
    ) -> LLMResult:
        """Generate n completions for one prompt through the gateway.

        For n>1 (e.g. ResponseRelevancy samples several reverse-questions) we nudge the
        temperature per sample so each is a DISTINCT gateway cache key — otherwise identical
        cached replies would collapse the diversity the metric depends on.
        """
        text = prompt.to_string()
        base = 0.0 if temperature is None else float(temperature)
        generations = []
        for i in range(n):
            temp = base if n == 1 else min(0.9, base + 0.2 * i + 0.1)
            out = self.gateway.complete(
                _JUDGE_SYSTEM, text, temperature=temp, max_tokens=self.max_tokens
            )
            generations.append(Generation(text=out, generation_info={"finish_reason": "stop"}))
        return LLMResult(generations=[generations])

    async def agenerate_text(
        self, prompt, n: int = 1, temperature=None, stop=None, callbacks=None
    ) -> LLMResult:
        """Async wrapper: run the (blocking, cached) gateway call off the event loop."""
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)


class BGERagasEmbeddings(BaseRagasEmbeddings):
    """RAGAS embeddings backed by our bge Embedder (free, local, no API — D-005).

    Uses the passage encoding (no bge query-instruction prefix) on both sides so
    ResponseRelevancy compares question-to-question in one symmetric space.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        super().__init__(cache=None)
        self.embedder = embedder or Embedder()

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self.embedder.encode_passages([text])[0]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.embedder.encode_passages(list(texts))
        return [[float(x) for x in v] for v in vecs]

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)
