"""Query decomposition: split a question into 1-3 focused search queries (Layer 5b-i).

Docs answers to real questions are scattered across pages (PLAN §0); a single dense/
BM25 query often misses one facet. Decomposing into sub-queries and retrieving each is
the intended fix for the real-slice weakness (D-031) — and it routes to Groq 8B, the
cheap high-volume model (D-008), since the agent may call it on every question. A
simple question yields one (cleaned) query, so decomposition degrades gracefully to
plain hybrid retrieval.
"""

from __future__ import annotations

import json

from src.config import get_settings

_MAX_SUBQUERIES = 3

_SYSTEM = (
    "You decide whether a LangChain/LangGraph documentation question must be split for "
    "retrieval. Be conservative:\n"
    "- DEFAULT: return the question UNCHANGED as a single query. Do not paraphrase it, "
    "add keywords, or generate variations — a single-topic question stays as one query.\n"
    "- ONLY if the question clearly asks about two or more DISTINCT things (e.g. 'what "
    "changed AND how do I migrate'), split it into 2-3 self-contained sub-queries.\n"
    'Return ONLY a JSON object: {"queries": ["...", "..."]}. No prose.'
)


def parse_subqueries(raw: str, fallback: str) -> list[str]:
    """Parse the model's JSON into a clean, capped, de-duplicated sub-query list.

    Any malformed/empty response falls back to the original question, so the retriever
    always has at least one query (degrades to plain hybrid retrieval).
    """
    try:
        queries = json.loads(raw).get("queries", [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        return [fallback]

    seen: dict[str, None] = {}  # preserve order, dedupe
    for q in queries:
        if isinstance(q, str) and q.strip():
            seen.setdefault(q.strip(), None)
    cleaned = list(seen)[:_MAX_SUBQUERIES]
    return cleaned or [fallback]


def decompose(question: str, gateway=None) -> list[str]:
    """Return 1-3 sub-queries for the question (Groq 8B via the LLM gateway)."""
    if gateway is None:
        from src.llm.gateway import LLMGateway

        gateway = LLMGateway(get_settings().cheap_model)
    raw = gateway.complete(_SYSTEM, f"Question: {question}", response_json=True, max_tokens=256)
    return parse_subqueries(raw, question)
