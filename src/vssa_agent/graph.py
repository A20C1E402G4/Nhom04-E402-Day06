"""LangGraph StateGraph wiring for the VSSA agent."""
from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import MAX_TOOL_TURNS, load_system_prompt
from .state import AgentState
from .tools import TOOLS

try:
    from telemetry.logger import log_failed_intent, log_node
except ImportError:  # pragma: no cover - telemetry is optional in tests
    def log_node(**_: Any) -> None: ...
    def log_failed_intent(_: str) -> None: ...


FALLBACK_VI: str = (
    "Thông tin này tôi chưa rõ. Hãy trao đổi với saler để được tư vấn kỹ hơn."
)


def _build_agent_node(llm: BaseChatModel):
    bound = llm.bind_tools(TOOLS)
    system_prompt = load_system_prompt()

    def agent_node(state: AgentState) -> dict[str, Any]:
        # Failure Mode #3: enforce Max-turn Limit.
        if state.get("tool_turns", 0) >= MAX_TOOL_TURNS:
            return {
                "messages": [AIMessage(content=FALLBACK_VI)],
            }

        messages = state["messages"]
        # Inject the system prompt only if the conversation has not already had one.
        has_system = any(
            getattr(m, "type", None) == "system" for m in messages
        )
        if not has_system:
            messages = [SystemMessage(content=system_prompt), *messages]

        # ── Data Flywheel: feed the learned user profile back into the LLM ──
        # The user_context dict is mutated by Streamlit slider interactions and
        # persisted via the checkpointer. Surfacing it here is what makes the
        # "lần chat sau AI tự động áp dụng mức vay này" loop in case_1.md work.
        ctx = state.get("user_context") or {}
        if ctx:
            profile_lines = "\n".join(f"  - {k}: {v}" for k, v in ctx.items())
            messages = [
                *messages,
                SystemMessage(
                    content=(
                        "<user_profile>\n"
                        f"{profile_lines}\n"
                        "</user_profile>\n"
                        "Áp dụng các sở thích đã học ở trên khi gọi tool "
                        "(ví dụ: nếu có loan_preference_pct, dùng giá trị đó "
                        "thay cho mặc định 70%)."
                    )
                ),
            ]

        started = time.perf_counter()
        response = bound.invoke(messages)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        tool_calls = getattr(response, "tool_calls", []) or []
        log_node(
            node_name="agent",
            elapsed_ms=elapsed_ms,
            tool_calls=[tc.get("name") for tc in tool_calls],
        )

        update: dict[str, Any] = {"messages": [response]}
        if tool_calls:
            update["tool_turns"] = state.get("tool_turns", 0) + 1
        return update

    return agent_node


def build_graph(
    llm: BaseChatModel,
    *,
    checkpointer: Any | None = None,
):
    """Compile and return the VSSA LangGraph.

    Pass an explicit `checkpointer` for production (SqliteSaver) or leave as
    None to fall back to an in-process MemorySaver (used by tests).
    """
    graph = StateGraph(AgentState)
    graph.add_node("agent", _build_agent_node(llm))
    graph.add_node("tools", ToolNode(TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer or MemorySaver())
