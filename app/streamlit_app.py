"""Streamlit demo for VSSA Case 1.

Run with::

    streamlit run app/streamlit_app.py

Implements `flow/case_1.md`:

* Happy Path  — chat → LangGraph agent → minimal VF 7 card.
* Loan path   — only revealed when the user asks (or clicks "Xem trả góp"),
                with a slider that recomputes locally (no LLM round-trip).
* Booking     — 3-step form (details → confirm → save) writing to
                ``logs/bookings.jsonl``.
* Flywheel    — slider corrections + booking leads + failed intents are logged,
                and the LLM receives a `<user_profile>` SystemMessage on every
                turn so it can apply learned preferences. State persists across
                browser reloads via SqliteSaver keyed by a thread_id stored in
                ``st.query_params``.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Make `src/` importable when launching via `streamlit run app/streamlit_app.py`.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from core.openai_provider import OpenAIProvider  # noqa: E402
from telemetry import logger as telemetry_logger  # noqa: E402
from telemetry import metrics as telemetry_metrics  # noqa: E402
from vssa_agent.config import CHECKPOINT_DB_PATH, LOGS_DIR  # noqa: E402
from vssa_agent.graph import build_graph  # noqa: E402
from vssa_agent.tools import _calculate_loan_impl, _find_showrooms_impl  # noqa: E402

st.set_page_config(page_title="VSSA — VinFast Smart Sales Agent", layout="wide")

DEFAULT_QUERY = "Tôi có 800 triệu, nhà 4 người, ở chung cư, nên mua xe nào?"
JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
LOAN_KEYWORDS = (
    "trả góp",
    "vay",
    "tài chính",
    "lãi suất",
    "trả trước",
    "góp",
    "khoản vay",
)
SKELETON_MD = "_⏳ VSSA đang soạn câu trả lời..._"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt_vnd(value: int | float | None) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}".replace(",", ".") + " ₫"


def fmt_slot(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return iso


def parse_agent_payload(text: str) -> dict[str, Any] | None:
    match = JSON_BLOCK_RE.search(text or "")
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def strip_json_block(text: str) -> str:
    return JSON_BLOCK_RE.sub("", text or "").strip()


def count_jsonl(filename: str) -> int:
    path = LOGS_DIR / filename
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def user_mentioned_loan(text: str) -> bool:
    lowered = (text or "").lower()
    return any(kw in lowered for kw in LOAN_KEYWORDS)


# ---------------------------------------------------------------------------
# LangGraph wiring (cached, persistent)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_compiled_graph():
    """Build the agent graph once with a SqliteSaver shared across reruns."""
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    saver = SqliteSaver(conn)
    return build_graph(OpenAIProvider().chat_model(), checkpointer=saver)


def get_or_create_thread_id() -> str:
    """Stable per-browser session id stored in URL query params.

    This is what makes cross-session memory work: refreshing the page keeps the
    same `thread_id`, so SqliteSaver returns the prior AgentState (including
    learned `user_context`).
    """
    if "thread_id" in st.query_params:
        return st.query_params["thread_id"]
    tid = str(uuid.uuid4())
    st.query_params["thread_id"] = tid
    return tid


def load_persisted_user_context(thread_id: str) -> dict[str, Any]:
    try:
        app = get_compiled_graph()
        snapshot = app.get_state({"configurable": {"thread_id": thread_id}})
        return dict((snapshot.values or {}).get("user_context") or {})
    except Exception:  # noqa: BLE001
        return {}


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = get_or_create_thread_id()
        telemetry_metrics.incr_session()
        seeded = load_persisted_user_context(st.session_state.session_id)
        st.session_state.user_context = seeded or {"loan_preference_pct": 70}
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_payload", None)
    st.session_state.setdefault("show_loan", False)
    st.session_state.setdefault(
        "loan_pct", int(st.session_state.user_context.get("loan_preference_pct", 70))
    )
    st.session_state.setdefault("loan_term", 96)
    st.session_state.setdefault("booking_step", "idle")
    st.session_state.setdefault("booking_draft", None)


def invoke_agent(user_text: str) -> str:
    app = get_compiled_graph()
    config = {"configurable": {"thread_id": st.session_state.session_id}}
    state_input = {
        "messages": [HumanMessage(content=user_text)],
        "user_context": st.session_state.user_context,
        "tool_turns": 0,
    }
    final_state = app.invoke(state_input, config=config)
    last: AIMessage = final_state["messages"][-1]
    return getattr(last, "content", "") or ""


# ---------------------------------------------------------------------------
# Sidebar — Data Flywheel visibility
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    with st.sidebar:
        st.header("🧠 Trí nhớ AI")
        st.caption("Sở thích AI đã học từ các lượt sửa lưng của bạn.")

        ctx = st.session_state.get("user_context") or {}
        if ctx:
            for key, value in ctx.items():
                st.markdown(f"- **{key}**: `{value}`")
        else:
            st.caption("_(chưa học được sở thích nào)_")

        st.divider()
        st.markdown("**📡 Tín hiệu Data Flywheel**")
        st.markdown(f"- Sửa lỗi (corrections): `{count_jsonl('corrections.jsonl')}`")
        st.markdown(f"- Lái thử đã đặt: `{count_jsonl('bookings.jsonl')}`")
        st.markdown(
            f"- Câu hỏi out-of-scope: `{count_jsonl('failed_intents.jsonl')}`"
        )
        st.caption(
            "Tín hiệu được ghi vào `logs/*.jsonl`. AI áp dụng `user_context` "
            "ở mỗi lượt thông qua `<user_profile>` SystemMessage."
        )

        st.divider()
        if st.button("🗑️ Xoá trí nhớ phiên này"):
            new_id = str(uuid.uuid4())
            st.query_params["thread_id"] = new_id
            for key in (
                "session_id",
                "user_context",
                "chat_history",
                "last_payload",
                "show_loan",
                "loan_pct",
                "loan_term",
                "booking_step",
                "booking_draft",
            ):
                st.session_state.pop(key, None)
            st.rerun()


# ---------------------------------------------------------------------------
# Right column — vehicle / loan / booking
# ---------------------------------------------------------------------------


def render_vehicle_card(vehicle: dict[str, Any]) -> None:
    st.subheader(f"🚗 {vehicle.get('model_name', '—')}")
    st.markdown(
        f"**Giá từ:** {fmt_vnd(vehicle.get('price_without_battery_vnd'))}  \n"
        f"**Quãng đường:** {vehicle.get('max_range_km', '—')} km / lần sạc"
    )
    features = vehicle.get("key_features") or []
    if features:
        st.markdown("\n".join(f"- {f}" for f in features[:4]))


def render_loan_section(vehicle: dict[str, Any]) -> None:
    if not st.session_state.show_loan:
        if st.button("💰 Xem phương án trả góp"):
            st.session_state.show_loan = True
            st.rerun()
        return

    vehicle_price = int(vehicle.get("price_without_battery_vnd") or 0)
    if vehicle_price <= 0:
        return

    with st.expander("Phương án trả góp", expanded=True):
        prev_pct = int(st.session_state.loan_pct)
        cols = st.columns(2)
        with cols[0]:
            new_pct = st.slider(
                "Tỷ lệ vay (%)", 0, 80, prev_pct, step=5, key="loan_pct_slider"
            )
        with cols[1]:
            term_options = [24, 36, 48, 60, 72, 84, 96]
            try:
                term_idx = term_options.index(int(st.session_state.loan_term))
            except ValueError:
                term_idx = len(term_options) - 1
            new_term = st.selectbox(
                "Kỳ hạn (tháng)", term_options, index=term_idx, key="loan_term_select"
            )

        result = _calculate_loan_impl(
            vehicle_price_vnd=vehicle_price,
            loan_percentage=int(new_pct),
            duration_months=int(new_term),
        )

        m1, m2 = st.columns(2)
        m1.metric("Trả trước", fmt_vnd(result["down_payment_vnd"]))
        m2.metric("Trả góp / tháng", fmt_vnd(result["monthly_payment_vnd"]))
        st.caption(result["disclaimer"])

        if int(new_pct) != prev_pct:
            telemetry_logger.log_correction(
                {
                    "field": "loan_percentage",
                    "from": prev_pct,
                    "to": int(new_pct),
                    "session_id": st.session_state.session_id,
                }
            )
            telemetry_metrics.incr_correction()
            st.session_state.user_context["loan_preference_pct"] = int(new_pct)
            st.session_state.loan_pct = int(new_pct)
        st.session_state.loan_term = int(new_term)


def _list_showrooms_for(model_id: str) -> list[dict[str, Any]]:
    result = _find_showrooms_impl(model_id=model_id)
    if not result.get("found"):
        return []
    return [result["recommended_showroom"], *result.get("alternatives", [])]


def render_booking_section(vehicle: dict[str, Any]) -> None:
    model_id = vehicle.get("model_id")
    if not model_id:
        return

    step = st.session_state.booking_step
    if step == "idle":
        if st.button("📅 Đặt lịch lái thử", type="primary"):
            st.session_state.booking_step = "form"
            st.rerun()
        return
    if step == "form":
        _render_booking_form(model_id)
        return
    if step == "confirm":
        _render_booking_confirm()
        return
    if step == "done":
        _render_booking_done()


def _render_booking_form(model_id: str) -> None:
    showrooms = _list_showrooms_for(model_id)
    if not showrooms:
        st.warning(
            "Hiện chưa có showroom hỗ trợ lái thử mẫu xe này. "
            "Hãy trao đổi với saler để được tư vấn kỹ hơn."
        )
        if st.button("Quay lại"):
            st.session_state.booking_step = "idle"
            st.rerun()
        return

    with st.form("booking_form", clear_on_submit=False):
        st.markdown("**Thông tin liên hệ**")
        name = st.text_input("Họ và tên *")
        phone = st.text_input("Số điện thoại *")

        st.markdown("**Địa điểm & thời gian**")
        sr_idx = st.selectbox(
            "Showroom bạn muốn lái thử *",
            options=list(range(len(showrooms))),
            format_func=lambda i: f"{showrooms[i]['showroom_name']} — {showrooms[i]['address']}",
        )
        chosen_sr = showrooms[sr_idx]
        slot = st.selectbox(
            "Khung giờ *",
            options=chosen_sr["all_upcoming_slots"],
            format_func=fmt_slot,
        )

        c1, c2 = st.columns(2)
        submit = c1.form_submit_button("Xem lại", type="primary")
        cancel = c2.form_submit_button("Huỷ")

    if cancel:
        st.session_state.booking_step = "idle"
        st.rerun()
    if submit:
        if not name.strip() or not phone.strip():
            st.error("Vui lòng nhập đầy đủ họ tên và số điện thoại.")
            return
        st.session_state.booking_draft = {
            "name": name.strip(),
            "phone": phone.strip(),
            "model_id": model_id,
            "showroom": chosen_sr,
            "slot": slot,
        }
        st.session_state.booking_step = "confirm"
        st.rerun()


def _render_booking_confirm() -> None:
    d = st.session_state.booking_draft or {}
    sr = d.get("showroom", {})
    st.info(
        "**Vui lòng xác nhận lịch lái thử**  \n\n"
        f"- Khách hàng: **{d.get('name', '—')}** — {d.get('phone', '—')}  \n"
        f"- Mẫu xe: **{d.get('model_id', '—')}**  \n"
        f"- Showroom: **{sr.get('showroom_name', '—')}**  \n"
        f"  {sr.get('address', '')}  \n"
        f"- Thời gian: **{fmt_slot(d.get('slot', ''))}**"
    )
    c1, c2 = st.columns(2)
    if c1.button("✅ Xác nhận", type="primary"):
        telemetry_logger.log_booking(
            {
                "name": d["name"],
                "phone": d["phone"],
                "model_id": d["model_id"],
                "showroom_id": sr.get("showroom_id"),
                "showroom_name": sr.get("showroom_name"),
                "showroom_address": sr.get("address"),
                "slot": d["slot"],
                "session_id": st.session_state.session_id,
            }
        )
        telemetry_metrics.incr_lead()
        telemetry_metrics.persist()
        st.session_state.booking_step = "done"
        st.rerun()
    if c2.button("Quay lại"):
        st.session_state.booking_step = "form"
        st.rerun()


def _render_booking_done() -> None:
    d = st.session_state.booking_draft or {}
    sr = d.get("showroom", {})
    st.success(
        f"✅ Đã ghi nhận lịch lái thử cho **{d.get('name')}**.  \n"
        f"Showroom **{sr.get('showroom_name')}** sẽ liên hệ qua **{d.get('phone')}** "
        f"trước **{fmt_slot(d.get('slot', ''))}**."
    )
    if st.button("Đặt lịch khác"):
        st.session_state.booking_step = "idle"
        st.session_state.booking_draft = None
        st.rerun()


# ---------------------------------------------------------------------------
# Chat panel (with scrollable history + skeleton placeholder)
# ---------------------------------------------------------------------------


def render_chat_history():
    """Render past messages inside a fixed-height scroll container."""
    container = st.container(height=520, border=True)
    with container:
        if not st.session_state.chat_history:
            st.caption(
                "👋 Chào bạn! Hãy mô tả ngân sách và nhu cầu — VSSA sẽ gợi ý "
                "mẫu xe phù hợp."
            )
        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(content)
    return container


def process_user_message(prompt: str, chat_container) -> None:
    """Append user msg, show skeleton, invoke agent, swap in real reply, rerun."""
    st.session_state.chat_history.append(("user", prompt))
    if user_mentioned_loan(prompt):
        st.session_state.show_loan = True

    placeholder = None
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown(SKELETON_MD)

    try:
        reply = invoke_agent(prompt)
    except Exception as exc:  # noqa: BLE001
        reply = f"⚠️ Lỗi gọi agent: `{exc}`"
        telemetry_logger.log_failed_intent(prompt, reason=str(exc))

    cleaned = strip_json_block(reply)
    if placeholder is not None:
        placeholder.markdown(cleaned)
    st.session_state.chat_history.append(("assistant", cleaned))

    payload = parse_agent_payload(reply)
    if payload:
        st.session_state.last_payload = payload
        if "loan" in payload:
            st.session_state.show_loan = True
            loan = payload["loan"]
            st.session_state.loan_pct = int(loan.get("loan_percentage", 70))
            st.session_state.loan_term = int(loan.get("duration_months", 96))

    st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("VSSA — VinFast Smart Sales Agent")
    st.caption("Trợ lý bán hàng VinFast — tư vấn xe & đặt lịch lái thử trong 1 phút.")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Tư vấn xe")
        chat_container = render_chat_history()

    payload = st.session_state.last_payload
    with right:
        if payload and payload.get("vehicle"):
            vehicle = payload["vehicle"]
            render_vehicle_card(vehicle)
            render_loan_section(vehicle)
            render_booking_section(vehicle)
        else:
            st.info("Hãy gửi câu hỏi tư vấn ở khung chat bên trái để bắt đầu.")

    # `st.chat_input` MUST live at top-level (outside columns) to be pinned to
    # the bottom of the viewport — Streamlit only docks it when not nested.
    prompt = st.chat_input("Bạn cần tư vấn gì?")
    if prompt:
        process_user_message(prompt, chat_container)
    elif not st.session_state.chat_history:
        # Render the sample-question prompt below the (empty) chat container.
        with left:
            if st.button(f'💬 Dùng câu hỏi mẫu: "{DEFAULT_QUERY}"'):
                process_user_message(DEFAULT_QUERY, chat_container)


main()
