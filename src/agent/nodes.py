"""Agent nodes: retrieve → synthesize (cited) → attach citations (Layer 5a).

Each node is a small pure-ish function of the state so the graph stays inspectable
and 5b/5c can slot decomposition/grading/verification between them. The synthesis
prompt is grounding-first: answer ONLY from the numbered context, cite every claim
with [n], and refuse rather than guess (refusals are not hallucinations — PLAN §6).
"""

from __future__ import annotations

from src.agent.citations import resolve_citations
from src.agent.state import AgentState
from src.config import get_settings
from src.retrieval.embedder import prepare_text

_REFUSAL = "I don't know based on the provided documentation."

_SYSTEM = (
    "You answer questions strictly from excerpts of the LangChain/LangGraph "
    "documentation provided as numbered context passages.\n"
    "RULES (follow exactly):\n"
    "1. Use ONLY information stated in the numbered passages. Never use outside or "
    "prior knowledge, even if you are certain of the answer.\n"
    f"2. If the passages do not contain the answer, reply with EXACTLY this sentence "
    f"and nothing else: {_REFUSAL}\n"
    "3. Cite the passage number(s) that directly support each claim, like [1] or [2]. "
    "Only cite a passage if its text actually states the claim; never attach citations "
    "to a sentence they do not support.\n"
    "4. Do NOT invent code, examples, argument values, method signatures, or any "
    "specifics not written verbatim in the passages. If a passage shows no code, do "
    "not fabricate an illustrative snippet — describe only what the passages state. "
    "If you include a code snippet, copy it EXACTLY from a passage: never substitute "
    "argument values, string literals, or keyword arguments (e.g. do not replace "
    "`messages` with an invented prompt string). Inventing any such specific is a "
    "violation of rule 1.\n"
    "5. Be concise and technical.\n"
    "A question about a topic unrelated to these passages (e.g. general trivia) is a "
    "case for rule 2 — refuse, do not answer from memory."
)

# Per-passage cap in the prompt: chunks are ~700 chars (well under this), so this only
# trims the rare long code chunk while bounding synthesis token cost.
_CTX_CHARS = 1200


def format_context(chunks) -> str:
    """Render retrieved chunks as a numbered [i] list with heading + URL + text."""
    blocks = []
    for i, h in enumerate(chunks, start=1):
        text = prepare_text(h.text, _CTX_CHARS)
        head = h.heading_path or "(no heading)"
        blocks.append(f"[{i}] {head}\n{h.source_url}\n{text}")
    return "\n\n".join(blocks)


def retrieve_node(retriever):
    """Factory: node that fills state['chunks'] via the (hybrid) retriever."""

    def _retrieve(state: AgentState) -> dict:
        k = get_settings().agent_context_k
        hits, _ = retriever.search(state["question"], top_k=k)
        return {"chunks": hits}

    return _retrieve


def synthesize_node(gateway, max_tokens: int = 1024):
    """Factory: node that synthesizes a cited answer from the context via the LLM.

    On a self-correction retry (Layer 5c), state['retry_feedback'] is prepended so the
    model knows its previous draft was ungrounded and must answer only from the passages
    or refuse — the retry re-synthesizes over the SAME context (recall isn't the problem;
    re-retrieval was measured-rejected, D-037), it re-grounds the generation (D-041).
    """

    def _synthesize(state: AgentState) -> dict:
        chunks = state["chunks"]
        if not chunks:
            return {"answer": _REFUSAL}
        user = f"Question: {state['question']}\n\nContext:\n{format_context(chunks)}"
        feedback = state.get("retry_feedback")
        if feedback:
            user = f"{feedback}\n\n{user}"
        answer = gateway.complete(_SYSTEM, user, max_tokens=max_tokens)
        return {"answer": answer.strip()}

    return _synthesize


_RETRY_FEEDBACK = (
    "Your previous answer made one or more claims that are NOT supported by the numbered "
    "passages. Answer again using ONLY what the passages state; drop any unsupported claim. "
    f"If the passages do not contain the answer, reply with EXACTLY: {_REFUSAL}"
)


def retry_node(state: AgentState) -> dict:
    """Charge one retry and stage corrective feedback for the next synthesis (Layer 5c)."""
    return {"retries": state.get("retries", 0) + 1, "retry_feedback": _RETRY_FEEDBACK}


def refuse_node(state: AgentState) -> dict:
    """Replace an ungrounded draft with the refusal once the retry budget is spent (5c)."""
    return {"answer": _REFUSAL, "grounded": False}


def route_after_verify(state: AgentState) -> str:
    """Conditional edge: grounded -> cite; else retry while budget remains, else refuse (5c)."""
    if state.get("grounded", True):
        return "cite"
    if state.get("retries", 0) < state.get("max_retries", 0):
        return "retry"
    return "refuse"


def cite_node(state: AgentState) -> dict:
    """Resolve [n] markers against the context; surface any hallucinated markers."""
    citations, invalid = resolve_citations(state.get("answer", ""), state.get("chunks", []))
    return {"citations": citations, "invalid_citations": invalid}
