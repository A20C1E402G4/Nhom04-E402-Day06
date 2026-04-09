"""End-to-end LangGraph tests using a stubbed chat model.

We don't hit the real OpenAI API. Instead we feed a deterministic queue of
`AIMessage` responses (with prebuilt tool_calls) into a fake chat model so we
can assert that the graph routes Happy / Booking paths correctly.
"""
from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from vssa_agent.graph import build_graph


class ScriptedChatModel(BaseChatModel):
    """Returns prebuilt AIMessages from a fixed queue. Ignores prompt content."""

    responses: list[AIMessage]
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = min(self.cursor, len(self.responses) - 1)
        self.cursor += 1
        msg = self.responses[idx]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs: Any):  # type: ignore[override]
        return self


def _happy_path_responses() -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_vehicle_data",
                    "args": {"model_id": "VF7"},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate_loan",
                    "args": {
                        "vehicle_price_vnd": 850_000_000,
                        "loan_percentage": 70,
                        "duration_months": 96,
                        "bank_id": "vietcombank",
                    },
                    "id": "call_2",
                }
            ],
        ),
        AIMessage(
            content=(
                "Tôi đề xuất VF 7 cho gia đình bạn.\n"
                "```json\n"
                '{"recommendation": "VF7", "vehicle": {"model_id": "VF7"}, '
                '"loan": {"vehicle_price_vnd": 850000000, "loan_percentage": 70, '
                '"duration_months": 96, "down_payment_vnd": 255000000, '
                '"monthly_payment_vnd": 8500000, "bank_name": "Vietcombank", '
                '"interest_rate_percent_per_year": 8.5}, "cta": "book_test_drive"}\n'
                "```"
            ),
        ),
    ]


def test_happy_path_invokes_expected_tool_sequence() -> None:
    llm = ScriptedChatModel(responses=_happy_path_responses())
    app = build_graph(llm)

    final_state = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Tôi có 800 triệu, nhà 4 người, ở chung cư, nên mua xe nào?"
                )
            ],
            "user_context": {},
            "tool_turns": 0,
        },
        config={"configurable": {"thread_id": "test-happy"}},
    )

    msgs = final_state["messages"]
    tool_names = [
        tc["name"]
        for m in msgs
        if isinstance(m, AIMessage)
        for tc in (getattr(m, "tool_calls", []) or [])
    ]
    assert tool_names == ["get_vehicle_data", "calculate_loan"]

    last = msgs[-1]
    assert isinstance(last, AIMessage)
    assert "VF 7" in last.content
    assert "```json" in last.content


def test_booking_path_invokes_find_showrooms() -> None:
    booking_responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "find_showrooms",
                    "args": {"model_id": "VF7", "district": "Thủ Đức"},
                    "id": "call_b1",
                }
            ],
        ),
        AIMessage(
            content=(
                "Đã tìm thấy showroom phù hợp.\n"
                "```json\n"
                '{"cta": "test_drive_booked", "showroom": '
                '{"showroom_name": "VinFast Thảo Điền", '
                '"address": "Vincom Mega Mall Thảo Điền, TP Thủ Đức", '
                '"next_slot": "2026-04-10T09:00:00"}}\n'
                "```"
            ),
        ),
    ]
    llm = ScriptedChatModel(responses=booking_responses)
    app = build_graph(llm)

    final_state = app.invoke(
        {
            "messages": [HumanMessage(content="Đăng ký lái thử VF 7")],
            "user_context": {},
            "tool_turns": 0,
        },
        config={"configurable": {"thread_id": "test-booking"}},
    )

    tool_names = [
        tc["name"]
        for m in final_state["messages"]
        if isinstance(m, AIMessage)
        for tc in (getattr(m, "tool_calls", []) or [])
    ]
    assert tool_names == ["find_showrooms"]
    assert "VinFast Thảo Điền" in final_state["messages"][-1].content
