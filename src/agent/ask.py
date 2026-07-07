"""CLI: ask the cited-answer agent a question (Layer 5a gate).

Run:  ./tasks.ps1 ask "how do I stream tokens from a chat model?"
Prints the synthesized answer with inline [n] markers, then the resolved sources
each marker points to, plus a warning if the model emitted any citation that does
not resolve to a retrieved chunk (a hallucinated citation).
"""

from __future__ import annotations

import argparse

from src.agent.graph import answer_question


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the cited-answer agent (Layer 5a).")
    ap.add_argument("question", nargs="+", help="the question to answer")
    args = ap.parse_args()
    question = " ".join(args.question)

    state = answer_question(question)
    print(f"\nQ: {question}\n")
    print(state.get("answer", "").strip())

    citations = state.get("citations", [])
    if citations:
        print("\nSources:")
        for c in citations:
            print(f"  [{c.marker}] {c.heading_path}")
            print(f"      {c.source_url}")

    invalid = state.get("invalid_citations", [])
    if invalid:
        print(f"\n[warning] unresolved citation markers (not in context): {invalid}")


if __name__ == "__main__":
    main()
