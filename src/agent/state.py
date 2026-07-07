"""LangGraph state for the cited-answer agent (Layer 5a, D-035).

Kept as a single typed channel dict so 5b/5c can add fields (sub-queries, grades,
grounding verdict) without reshaping the graph. Hit/Citation are carried as plain
objects between nodes; only the CLI/serialization layer renders them.
"""

from typing import TypedDict

# Imported at runtime (not TYPE_CHECKING): LangGraph resolves the state's annotations
# via get_type_hints() when the StateGraph is built, so the names must exist at runtime.
from src.agent.citations import Citation
from src.retrieval.search import Hit


class AgentState(TypedDict, total=False):
    """Channels threaded through the graph (total=False: nodes fill them in turn)."""

    question: str
    chunks: list[Hit]           # retrieved context passages, numbered [1..k]
    answer: str                 # synthesized answer text with inline [n] markers
    citations: list[Citation]   # resolved [n] -> source
    invalid_citations: list[int]  # [n] markers pointing outside the context (hallucinated)
