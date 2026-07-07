"""Grounding verification: check the drafted answer against its cited passages (Layer 5b).

The measured fix for D-036: *pre*-synthesis relevance grading could not catch strong-prior
hallucination ("capital of France?" → "Paris") because every judgment made before the
answer exists is contaminated by the model's certainty it knows the answer. A *post*-
synthesis check of the specific claim against the specific passage text is far more
constrained — and, counter-intuitively, **Groq 8B verifies better than Gemini here**:
not "knowing" Paris, it actually reads the passages and finds the claim absent (D-038).
So verification routes to the cheap Groq model (D-008), which is both correct and cheap.

Fails OPEN: a malformed verdict trusts the answer, so a verifier glitch never wrongly
refuses a good answer.
"""

from __future__ import annotations

import json
import logging

from src.agent.nodes import _REFUSAL, format_context
from src.config import get_settings

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You verify whether an ANSWER is grounded in the numbered PASSAGES it draws from. "
    "Check every factual claim in the answer against the passage text.\n"
    "- The answer is grounded ONLY if each claim is directly supported by the passages. "
    "Do not use your own knowledge — judge solely from the passage text.\n"
    "- If any claim is not stated in the passages, the answer is NOT grounded.\n"
    'Return ONLY JSON: {"grounded": true} or {"grounded": false}.'
)

_FALSE_WORDS = {"false", "no", "0", "not grounded"}


def parse_grounded(raw: str) -> bool:
    """Parse the verifier's verdict. Malformed/missing → True (fail open, trust answer)."""
    try:
        val = json.loads(raw)["grounded"]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return True
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() not in _FALSE_WORDS
    return True


def verify_grounded(question: str, answer: str, chunks, gateway) -> bool:
    """Ask the verifier whether the answer's claims are supported by the passages."""
    user = f"ANSWER: {answer}\n\nPASSAGES:\n{format_context(chunks)}"
    raw = gateway.complete(_SYSTEM, user, response_json=True, max_tokens=64)
    return parse_grounded(raw)


def verify_node(gateway):
    """Factory: node that refuses the answer if it isn't grounded in the passages (Groq)."""

    def _verify(state) -> dict:
        answer = state.get("answer", "").strip()
        chunks = state.get("chunks", [])
        # Nothing to verify if the model already refused or there is no context.
        if not chunks or not answer or answer == _REFUSAL:
            return {"grounded": True}
        grounded = verify_grounded(state["question"], answer, chunks, gateway)
        if not grounded:
            logger.info("verify: answer not grounded -> refusing")
            return {"grounded": False, "answer": _REFUSAL}
        return {"grounded": True}

    return _verify


def default_verifier():
    """The cheap Groq gateway used for grounding verification (D-008/D-038)."""
    from src.llm.gateway import LLMGateway

    return LLMGateway(get_settings().cheap_model)
