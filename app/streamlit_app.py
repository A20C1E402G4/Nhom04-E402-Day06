"""Streamlit demo for VSSA Case 1 — single-column, inline-action chat UI.

Run with::

    streamlit run app/streamlit_app.py

Design highlights
-----------------
* **One column** — the chat *is* the UI. Vehicle cards, the loan toggle, and
  the booking CTA all render inline inside the latest assistant
  `st.chat_message`.
* **Loan slider with explicit confirm** — dragging only previews the numbers.
  A correction is logged to ``logs/corrections.jsonl`` (and written back into
  ``user_context``) only when the user clicks **✅ Xác nhận thay đổi**.
* **Agent-driven booking** — clicking **📅 Đặt lịch lái thử** injects a
  user message into the chat. The LLM then walks the customer through
  location → showroom suggestion → name/phone → `book_test_drive`. No
  rule-based form.
* **Flywheel** — SqliteSaver persists AgentState across reloads;
  ``user_context`` is injected back into the LLM as a ``<user_profile>``
  SystemMessage on every turn.
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

from core.local_provider import LocalProvider  # noqa: E402
from core.openai_provider import OpenAIProvider  # noqa: E402
from telemetry import logger as telemetry_logger  # noqa: E402
from telemetry import metrics as telemetry_metrics  # noqa: E402
from vssa_agent.config import CHECKPOINT_DB_PATH, LOGS_DIR  # noqa: E402
from vssa_agent.graph import build_graph  # noqa: E402
from vssa_agent.tools import _calculate_loan_impl  # noqa: E402

st.set_page_config(page_title="VSSA — VinFast Smart Sales Agent", layout="centered")

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
TERM_OPTIONS = [24, 36, 48, 60, 72, 84, 96]


# ---------------------------------------------------------------------------
# Pure helpers
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


# Available models — Local Qwen 2.5 first (default)
MODEL_OPTIONS: dict[str, dict] = {
    "🏠 Local — Qwen 2.5 (Ollama)": {"provider": "local", "model": "qwen2.5"},
    "🏠 Local — Qwen 2.5 14B (Ollama)": {"provider": "local", "model": "qwen2.5:14b"},
    "🏠 Local — Llama 3 8B (Ollama)": {"provider": "local", "model": "llama3:8b"},
    "☁️ OpenAI — GPT-4o": {"provider": "openai", "model": None},
}


def _make_provider(label: str):
    cfg = MODEL_OPTIONS[label]
    if cfg["provider"] == "local":
        return LocalProvider(model=cfg["model"])
    return OpenAIProvider()


@st.cache_resource(show_spinner=False)
def get_compiled_graph(_model_label: str):
    """Build the agent graph once with a SqliteSaver shared across reruns."""
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    saver = SqliteSaver(conn)
    provider = _make_provider(_model_label)
    return build_graph(provider.chat_model(), checkpointer=saver)


def get_or_create_thread_id() -> str:
    if "thread_id" in st.query_params:
        return st.query_params["thread_id"]
    tid = str(uuid.uuid4())
    st.query_params["thread_id"] = tid
    return tid


def load_persisted_user_context(thread_id: str) -> dict[str, Any]:
    try:
        label = st.session_state.get("model_label", list(MODEL_OPTIONS.keys())[0])
        app = get_compiled_graph(label)
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
    # chat_history entries: {"role", "content", "payload": dict|None}
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_vehicle", None)
    st.session_state.setdefault("show_loan", False)
    st.session_state.setdefault("loan_term", 96)
    st.session_state.setdefault("pending_user_input", None)
    st.session_state.setdefault("agent_status", "")


def invoke_agent(user_text: str):
    label = st.session_state.get("model_label", list(MODEL_OPTIONS.keys())[0])
    app = get_compiled_graph(label)
    config = {"configurable": {"thread_id": st.session_state.session_id}}
    state_input = {
        "messages": [HumanMessage(content=user_text)],
        "user_context": st.session_state.user_context,
        "tool_turns": 0,
    }
    
    final_content = ""
    # Use updates mode to see which node is running
    for chunk in app.stream(state_input, config=config, stream_mode="updates"):
        for node_name, output in chunk.items():
            if node_name == "agent":
                msg = output["messages"][-1]
                if getattr(msg, "tool_calls", None):
                    tool_names = [tc["name"] for tc in msg.tool_calls]
                    readable_names = {
                        "get_vehicle_data": "lấy thông số xe",
                        "calculate_loan": "tính toán khoản vay",
                        "find_showrooms": "tìm kiếm showroom",
                        "get_promotion": "tra cứu khuyến mãi",
                        "book_test_drive": "ghi nhận lịch hẹn",
                        "find_charging_stations": "tìm kiếm trạm sạc"
                    }
                    display_tools = [readable_names.get(n, n) for n in tool_names]
                    yield f"🤖 Agent đang {', '.join(display_tools)}..."
                else:
                    yield "✍️ Agent đang soạn câu trả lời..."
            elif node_name == "tools":
                yield "🛠️ Tools đang thực thi..."
            
            # Keep track of the last state to get the final message
            if "messages" in output:
                final_content = output["messages"][-1].content

    yield final_content


# ---------------------------------------------------------------------------
# Sidebar — Data Flywheel visibility
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    with st.sidebar:
        # ── Model selector ──
        st.header("⚙️ Chọn mô hình AI")
        model_labels = list(MODEL_OPTIONS.keys())
        current = st.session_state.get("model_label", model_labels[0])
        try:
            idx = model_labels.index(current)
        except ValueError:
            idx = 0
        chosen = st.selectbox(
            "Model", model_labels, index=idx, key="_model_selector"
        )
        if chosen != st.session_state.get("model_label"):
            st.session_state.model_label = chosen
            st.rerun()

        st.divider()
        st.header("⚡ Trạng thái Agent")
        # Fixed placeholder for real-time status updates from anywhere
        st.session_state.sidebar_status_area = st.empty()
        status = st.session_state.get("agent_status", "")
        if status:
            st.session_state.sidebar_status_area.info(status)
        else:
            st.session_state.sidebar_status_area.caption("Agent đang sẵn sàng.")
        
        st.divider()
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
            for key in list(st.session_state.keys()):
                if key.startswith(("loan_pct_", "loan_term_")):
                    st.session_state.pop(key, None)
            for key in (
                "session_id",
                "user_context",
                "chat_history",
                "last_vehicle",
                "show_loan",
                "loan_term",
                "pending_user_input",
            ):
                st.session_state.pop(key, None)
            st.rerun()


# ---------------------------------------------------------------------------
# Inline action widgets rendered inside assistant chat_message
# ---------------------------------------------------------------------------


def render_vehicle_summary(vehicle: dict[str, Any]) -> None:
    st.markdown(
        f"**🚗 {vehicle.get('model_name', '—')}**  \n"
        f"Giá từ: **{fmt_vnd(vehicle.get('price_without_battery_vnd'))}** · "
        f"Quãng đường: **{vehicle.get('max_range_km', '—')} km**"
    )
    features = vehicle.get("key_features") or []
    if features:
        st.markdown("\n".join(f"- {f}" for f in features[:4]))


def render_loan_inline(vehicle: dict[str, Any], turn_key: str) -> None:
    """Loan expander with draft slider + explicit confirm button."""
    price = int(vehicle.get("price_without_battery_vnd") or 0)
    if price <= 0:
        return

    committed_pct = int(st.session_state.user_context.get("loan_preference_pct", 70))
    slider_key = f"loan_pct_{turn_key}"
    term_key = f"loan_term_{turn_key}"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = committed_pct
    if term_key not in st.session_state:
        st.session_state[term_key] = int(st.session_state.loan_term)

    with st.expander("💰 Phương án trả góp (nháp — chưa lưu)", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            draft_pct = st.slider(
                "Tỷ lệ vay (%)", 0, 80, step=5, key=slider_key
            )
        with cols[1]:
            try:
                term_idx = TERM_OPTIONS.index(int(st.session_state[term_key]))
            except ValueError:
                term_idx = len(TERM_OPTIONS) - 1
            draft_term = st.selectbox(
                "Kỳ hạn (tháng)",
                TERM_OPTIONS,
                index=term_idx,
                key=term_key,
            )

        result = _calculate_loan_impl(
            vehicle_price_vnd=price,
            loan_percentage=int(draft_pct),
            duration_months=int(draft_term),
        )
        m1, m2 = st.columns(2)
        m1.metric("Trả trước", fmt_vnd(result["down_payment_vnd"]))
        m2.metric("Trả góp / tháng", fmt_vnd(result["monthly_payment_vnd"]))
        st.caption(result["disclaimer"])

        dirty = int(draft_pct) != committed_pct or int(draft_term) != int(
            st.session_state.loan_term
        )
        if dirty:
            st.warning(
                f"Bạn đang xem thử mức **{draft_pct}% / {draft_term} tháng**. "
                "Nhấn xác nhận để VSSA ghi nhớ làm mặc định cho các lượt sau."
            )
            if st.button("✅ Xác nhận thay đổi", key=f"confirm_{turn_key}"):
                telemetry_logger.log_correction(
                    {
                        "field": "loan_percentage",
                        "from": committed_pct,
                        "to": int(draft_pct),
                        "term_from": int(st.session_state.loan_term),
                        "term_to": int(draft_term),
                        "session_id": st.session_state.session_id,
                    }
                )
                telemetry_metrics.incr_correction()
                st.session_state.user_context["loan_preference_pct"] = int(draft_pct)
                st.session_state.loan_term = int(draft_term)
                st.toast("Đã ghi nhận. AI sẽ áp dụng từ lượt sau.", icon="✅")
                st.rerun()
        else:
            st.caption("_Mức hiện tại trùng với trí nhớ đã lưu._")


def render_charging_map(stations: list[dict[str, Any]]) -> None:
    """Render a map with charging station locations."""
    if not stations:
        return
    
    import pandas as pd
    map_data = []
    for s in stations:
        name = s.get("location_name") or s.get("name") or s.get("station_name") or "Trạm sạc VinFast"
        coords = s.get("coordinates") or s
        lat = coords.get("lat")
        lon = coords.get("lon") or coords.get("lng")
        
        if lat is not None and lon is not None:
            map_data.append({
                "name": name,
                "lat": float(lat),
                "lon": float(lon)
            })
    
    if not map_data:
        st.warning("Không có dữ liệu tọa độ để hiển thị bản đồ.")
        return

    df = pd.DataFrame(map_data)
    st.subheader("📍 Bản đồ trạm sạc")
    st.map(df)
    for s in stations:
        name = s.get("location_name") or s.get("name") or "Trạm sạc"
        addr = s.get("address") or s.get("addr") or "Đang cập nhật địa chỉ"
        types = s.get("charger_types") or []
        st.caption(f"- **{name}**: {addr} ({', '.join(types)})")


def render_inline_actions(vehicle: dict[str, Any], turn_key: str) -> None:
    """Render the CTA buttons + loan expander for the LATEST vehicle payload."""
    render_vehicle_summary(vehicle)

    col_loan, col_book = st.columns(2)
    with col_loan:
        if not st.session_state.show_loan:
            if st.button(
                "💰 Xem phương án trả góp", key=f"show_loan_{turn_key}"
            ):
                st.session_state.show_loan = True
                st.rerun()
    with col_book:
        if st.button(
            "📅 Đặt lịch lái thử",
            type="primary",
            key=f"book_{turn_key}",
        ):
            model_id = vehicle.get("model_id", "")
            st.session_state.pending_user_input = (
                f"Tôi muốn đặt lịch lái thử {model_id}."
            )
            st.rerun()

    if st.session_state.show_loan:
        render_loan_inline(vehicle, turn_key)


# ---------------------------------------------------------------------------
# Chat rendering + turn processing
# ---------------------------------------------------------------------------


def _latest_vehicle_turn_idx() -> int:
    """Index of the most recent assistant entry whose payload has a vehicle."""
    for i in range(len(st.session_state.chat_history) - 1, -1, -1):
        entry = st.session_state.chat_history[i]
        if entry["role"] == "assistant":
            payload = entry.get("payload") or {}
            if payload.get("vehicle"):
                return i
    return -1


def render_chat(container) -> None:
    latest_vehicle_idx = _latest_vehicle_turn_idx()
    with container:
        if not st.session_state.chat_history:
            st.caption(
                "👋 Chào bạn! Hãy mô tả ngân sách và nhu cầu — VSSA sẽ gợi ý "
                "mẫu xe phù hợp."
            )
            if st.button(f'💬 Dùng câu hỏi mẫu'):
                st.session_state.pending_user_input = DEFAULT_QUERY
                st.rerun()

        for idx, entry in enumerate(st.session_state.chat_history):
            with st.chat_message(entry["role"]):
                if entry["content"]:
                    st.markdown(entry["content"])

                # Inline CTAs attach ONLY to the most recent vehicle
                # recommendation, so old turns stay static.
                if entry["role"] == "assistant" and idx == latest_vehicle_idx:
                    vehicle = (entry.get("payload") or {}).get("vehicle")
                    if vehicle:
                        st.session_state.last_vehicle = vehicle
                        render_inline_actions(vehicle, turn_key=f"turn{idx}")

                # Booking confirmation styled success message.
                # Assistant-specific inline content
                if entry["role"] == "assistant":
                    payload = entry.get("payload") or {}
                    
                    # Charging stations map
                    data = payload.get("charging_stations")
                    if data:
                        stations = []
                        if isinstance(data, list):
                            stations = data
                        elif isinstance(data, dict):
                            stations = data.get("stations", [])
                        render_charging_map(stations)

                    # Booking confirmation
                    if payload.get("cta") == "test_drive_booked":
                        sr = payload.get("showroom") or {}
                        st.success(
                            f"✅ Đã chốt lịch lái thử tại "
                            f"**{sr.get('showroom_name', '—')}** "
                            f"({sr.get('address', '')}) "
                            f"lúc **{fmt_slot(sr.get('slot', ''))}**."
                        )


def process_user_message(prompt: str, container, sidebar_status=None) -> None:
    """Append user msg, show skeleton, invoke agent, swap in real reply, rerun."""
    st.session_state.chat_history.append(
        {"role": "user", "content": prompt, "payload": None}
    )
    if user_mentioned_loan(prompt):
        st.session_state.show_loan = True

    placeholder = None
    with container:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown(SKELETON_MD)

    try:
        reply = ""
        # The generator yields status first, then final content
        for status_or_content in invoke_agent(prompt):
            if status_or_content.startswith(("🤖", "🛠️", "✍️")):
                st.session_state.agent_status = status_or_content
                placeholder.markdown(f"_{status_or_content}_")
                # Update sidebar status in real-time using the placeholder
                sb_area = st.session_state.get("sidebar_status_area")
                if sb_area:
                    sb_area.info(status_or_content)
            else:
                reply = status_or_content
        
        st.session_state.agent_status = "" # Clear status when done
    except Exception as exc:  # noqa: BLE001
        reply = f"⚠️ Lỗi gọi agent: `{exc}`"
        telemetry_logger.log_failed_intent(prompt, reason=str(exc))

    cleaned = strip_json_block(reply)
    if placeholder is not None:
        placeholder.markdown(cleaned or "_(không có phản hồi)_")

    payload = parse_agent_payload(reply)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": cleaned, "payload": payload}
    )

    if payload and payload.get("vehicle"):
        st.session_state.last_vehicle = payload["vehicle"]
    if payload and "loan" in payload:
        st.session_state.show_loan = True
        loan = payload["loan"]
        st.session_state.loan_term = int(loan.get("duration_months", 96))

    st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("VSSA — VinFast Smart Sales Agent")
    st.caption(
        "Trợ lý bán hàng VinFast — tư vấn xe, tính trả góp & đặt lịch lái thử."
    )

    chat_container = st.container(height=600, border=True)
    render_chat(chat_container)

    # `st.chat_input` MUST live at top level (outside any column) to be docked
    # to the bottom of the viewport — Streamlit only pins it when not nested.
    typed = st.chat_input("Bạn cần tư vấn gì?")
    if typed:
        st.session_state.pending_user_input = typed
        st.rerun()

    # Drain any queued input (from chat_input OR inline buttons).
    pending = st.session_state.pop("pending_user_input", None)
    if pending:
        process_user_message(
            pending, 
            chat_container, 
            sidebar_status=st.session_state.get("sidebar_status_placeholder")
        )


main()
