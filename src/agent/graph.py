"""Assemble the Layer 5a cited-answer graph: retrieve → synthesize → cite (D-035).

Linear for 5a, but built as a LangGraph StateGraph so 5b/5c add decomposition,
grading, rewrite, and grounding-verification as nodes/edges without a rewrite — and
so every node renders as a span once Langfuse is wired (Layer 7). Retriever and
synthesis gateway are injected (default: the deployed hybrid retriever + Gemini),
so the wiring is unit-testable with fakes and the retriever choice stays config/D-031.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import cite_node, retrieve_node, synthesize_node
from src.agent.state import AgentState
from src.agent.verify import verify_node
from src.config import get_settings


def build_agent(retriever=None, gateway=None, verifier=None):
    """Compile the retrieve→synthesize→verify→cite graph (injectable deps for testing).

    verify checks the drafted answer against its cited passages and refuses if it isn't
    grounded — the measured fix for hallucination on strong priors (D-036/D-038).
    retriever + synthesis gateway default to the deployed hybrid + Gemini; the verifier
    defaults to Groq 8B, which verifies better than Gemini here precisely because it does
    not "know" the answer and so actually reads the passages (D-038).
    """
    if retriever is None:
        from src.retrieval.search import HybridRetriever

        retriever = HybridRetriever()
    if gateway is None:
        from src.llm.gateway import LLMGateway

        gateway = LLMGateway(get_settings().synthesis_model)
    if verifier is None:
        from src.agent.verify import default_verifier

        verifier = default_verifier()

    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node(retriever))
    g.add_node("synthesize", synthesize_node(gateway))
    g.add_node("verify", verify_node(verifier))
    g.add_node("cite", cite_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", "verify")
    g.add_edge("verify", "cite")
    g.add_edge("cite", END)
    return g.compile()


def answer_question(question: str, *, retriever=None, gateway=None, verifier=None) -> AgentState:
    """Run the agent once and return the final state (answer + resolved citations)."""
    agent = build_agent(retriever=retriever, gateway=gateway, verifier=verifier)
    return agent.invoke({"question": question})
