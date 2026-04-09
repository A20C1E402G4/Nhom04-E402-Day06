"""CLI smoke test for the VSSA agent.

Usage::

    python -m vssa_agent "Tôi có 800 triệu, nhà 4 người, ở chung cư, nên mua xe nào?"
"""
from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage

from core.openai_provider import OpenAIProvider

from .graph import build_graph
from .state import AgentState


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python -m vssa_agent <user query>", file=sys.stderr)
        return 1

    query = " ".join(argv[1:])
    llm = OpenAIProvider().chat_model()
    app = build_graph(llm)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "user_context": {},
        "tool_turns": 0,
    }
    config = {"configurable": {"thread_id": "cli-smoke"}}
    final_state = app.invoke(initial_state, config=config)

    last = final_state["messages"][-1]
    print(getattr(last, "content", str(last)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
