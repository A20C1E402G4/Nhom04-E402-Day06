"""LangGraph AgentState definition for VSSA."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class UserContext(TypedDict, total=False):
    """Slot-style memory captured from the conversation."""

    budget_vnd: int
    family_size: int
    housing: str  # "apartment" | "house"
    loan_preference_pct: int
    preferred_model: str  # "VF5" | "VF7" | "VF8"
    location_hint: str


class AgentState(TypedDict):
    """Top-level state passed between LangGraph nodes."""

    messages: Annotated[list[AnyMessage], add_messages]
    user_context: UserContext
    tool_turns: int
