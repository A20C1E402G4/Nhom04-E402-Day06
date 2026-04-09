from __future__ import annotations

import json
import re
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Make `src/` importable
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from core.local_provider import LocalProvider  # noqa: E402
from core.openai_provider import OpenAIProvider  # noqa: E402
from telemetry import logger as telemetry_logger  # noqa: E402
from telemetry import metrics as telemetry_metrics  # noqa: E402
from vssa_agent.config import CHECKPOINT_DB_PATH, LOGS_DIR  # noqa: E402
from vssa_agent.graph import build_graph  # noqa: E402
from vssa_agent.tools import _calculate_loan_impl  # noqa: E402

st.set_page_config(
    page_title="VSSA — VinFast Smart Sales Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling — theme-agnostic (light & dark friendly)
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
        max-width: 900px;
    }
    /* .stApp intentionally omitted — let Streamlit's built-in theme control
       background and text colours so light mode works correctly. */
    [data-testid="stChatMessageUser"] {
        background-color: var(--secondary-background-color);
        border-radius: 15px 15px 0 15px;
        padding: 12px;
        margin-bottom: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    [data-testid="stChatMessageAssistant"] {
        background-color: transparent;
        margin-bottom: 12px;
    }
    .vssa-header {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .vssa-header h1 {
        color: #0088cc !important;
        font-family: 'Calibri', sans-serif !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        text-transform: uppercase;
    }
    /* Secondary / feedback buttons */
    .stButton button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        color: var(--text-color) !important;
        opacity: 0.7;
        font-size: 0.8rem !important;
        padding: 2px 10px !important;
        border-radius: 8px !important;
        height: auto !important;
        white-space: nowrap !important;
    }
    .stButton button[kind="secondary"]:hover {
        opacity: 1;
        border-color: #0088cc !important;
        color: #0088cc !important;
    }
    .stButton button[kind="primary"] { opacity: 1.0 !important; }
    /* Suggestion chips */
    div[data-testid="stVerticalBlock"] > div:nth-last-child(2) .stButton button {
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        font-size: 0.85rem !important;
        border-radius: 12px !important;
    }
    /* st.status */
    div[data-testid="stStatusWidget"] {
        background-color: rgba(0, 136, 204, 0.05) !important;
        border: 1px solid rgba(0, 136, 204, 0.2) !important;
        border-radius: 12px !important;
    }
    .stMetric {
        background-color: var(--secondary-background-color);
        padding: 10px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_QUERY = "Tôi có 800 triệu, nhà 4 người, ở chung cư, nên mua xe nào?"
JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
SKELETON_MD = "_⏳ VSSA đang soạn câu trả lời..._"
TERM_OPTIONS = [24, 36, 48, 60, 72, 84, 96]
LOAN_KEYWORDS = ("trả góp", "vay", "tài chính", "lãi suất", "trả trước", "góp", "khoản vay")

# Human-readable labels for tool calls (NhatVi)
TOOL_LABELS: dict[str, str] = {
    "get_vehicle_data": "lấy thông số xe",
    "get_all_vehicles": "tra cứu danh mục xe",
    "calculate_loan": "tính toán khoản vay",
    "find_showrooms": "tìm kiếm showroom",
    "get_promotion": "tra cứu khuyến mãi",
    "book_test_drive": "ghi nhận lịch hẹn",
    "find_charging_stations": "tìm kiếm trạm sạc",
}

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
    except Exception:
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


def log_feedback(sentiment: str, content: str) -> None:
    log_path = LOGS_DIR / "feedback.jsonl"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": st.session_state.get("session_id", "unknown"),
        "sentiment": sentiment,
        "content": content[:200] + "..." if len(content) > 200 else content,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def user_mentioned_loan(text: str) -> bool:
    lowered = (text or "").lower()
    return any(kw in lowered for kw in LOAN_KEYWORDS)


# ---------------------------------------------------------------------------
# Model switching (NhatVi) — OpenAI is default, local requires Ollama
# ---------------------------------------------------------------------------

MODEL_OPTIONS: dict[str, dict] = {
    "☁️ OpenAI — GPT-4o": {"provider": "openai", "model": None},
    "🏠 Local — Qwen 2.5 (Ollama)": {"provider": "local", "model": "qwen2.5"},
    "🏠 Local — Qwen 2.5 14B (Ollama)": {"provider": "local", "model": "qwen2.5:14b"},
    "🏠 Local — Llama 3 8B (Ollama)": {"provider": "local", "model": "llama3:8b"},
}
_DEFAULT_MODEL = "☁️ OpenAI — GPT-4o"


def _make_provider(label: str):
    cfg = MODEL_OPTIONS[label]
    if cfg["provider"] == "local":
        return LocalProvider(model=cfg["model"])
    return OpenAIProvider()


# ---------------------------------------------------------------------------
# LangGraph wiring (persistent across browser reloads via SqliteSaver)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_compiled_graph(_model_label: str):
    """Build the agent graph once; keyed on model label so switching recompiles."""
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
        label = st.session_state.get("model_label", _DEFAULT_MODEL)
        app = get_compiled_graph(label)
        snapshot = app.get_state({"configurable": {"thread_id": thread_id}})
        return dict((snapshot.values or {}).get("user_context") or {})
    except Exception:  # noqa: BLE001
        return {}


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        tid = get_or_create_thread_id()
        st.session_state.session_id = tid
        telemetry_metrics.incr_session()
        seeded = load_persisted_user_context(tid)
        st.session_state.user_context = seeded or {"loan_preference_pct": 70}
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_vehicle", None)
    st.session_state.setdefault("loan_term", 96)
    st.session_state.setdefault("show_loan", False)
    st.session_state.setdefault("pending_input", None)
    st.session_state.setdefault("feedback_map", {})
    # NhatVi: live status string displayed in sidebar while agent streams
    st.session_state.setdefault("agent_status", "")


# ---------------------------------------------------------------------------
# Agent invocation — streaming generator (NhatVi)
# ---------------------------------------------------------------------------


def invoke_agent(user_text: str):
    """Generator: yields human-readable status strings, then the final reply."""
    label = st.session_state.get("model_label", _DEFAULT_MODEL)
    app = get_compiled_graph(label)
    config = {"configurable": {"thread_id": st.session_state.session_id}}
    state_input = {
        "messages": [HumanMessage(content=user_text)],
        "user_context": st.session_state.user_context,
        "tool_turns": 0,
    }

    final_content = ""
    for chunk in app.stream(state_input, config=config, stream_mode="updates"):
        for node_name, output in chunk.items():
            if node_name == "agent":
                msg = output["messages"][-1]
                if getattr(msg, "tool_calls", None):
                    names = [
                        TOOL_LABELS.get(tc["name"], tc["name"])
                        for tc in msg.tool_calls
                    ]
                    yield f"🤖 Agent đang {', '.join(names)}..."
                else:
                    yield "✍️ Agent đang soạn câu trả lời..."
            elif node_name == "tools":
                yield "🛠️ Tools đang thực thi..."
            if "messages" in output:
                final_content = output["messages"][-1].content or ""

    yield final_content


# ---------------------------------------------------------------------------
# process_user_message — replaces the old is_processing flag pattern (NhatVi)
# ---------------------------------------------------------------------------


def process_user_message(prompt: str, container) -> None:
    """Append user message, show live streaming status, swap in final reply."""
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
        for status_or_content in invoke_agent(prompt):
            if status_or_content.startswith(("🤖", "🛠️", "✍️")):
                st.session_state.agent_status = status_or_content
                placeholder.markdown(f"_{status_or_content}_")
                # Update sidebar status placeholder in real-time
                sb_area = st.session_state.get("sidebar_status_area")
                if sb_area:
                    sb_area.info(status_or_content)
            else:
                reply = status_or_content
        st.session_state.agent_status = ""
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
        st.session_state.loan_term = int(
            payload["loan"].get("duration_months", 96)
        )

    st.rerun()


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------


def render_loan_sidebar(vehicle: dict[str, Any]) -> None:
    """Loan calculator widget in the sidebar (dev's confirmed-save pattern)."""
    st.markdown("**💰 Dự tính trả góp**")
    price = int(vehicle.get("price_without_battery_vnd") or 0)
    if price <= 0:
        return
    committed_pct = int(st.session_state.user_context.get("loan_preference_pct", 70))
    committed_term = int(st.session_state.loan_term)
    draft_pct = st.slider("Tỷ lệ vay (%)", 0, 80, value=committed_pct, step=5)
    try:
        t_idx = TERM_OPTIONS.index(committed_term)
    except (ValueError, IndexError):
        t_idx = len(TERM_OPTIONS) - 1
    draft_term = st.selectbox("Kỳ hạn (tháng)", TERM_OPTIONS, index=t_idx)
    res = _calculate_loan_impl(price, int(draft_pct), int(draft_term))
    st.info(f"Trả trước: {fmt_vnd(res['down_payment_vnd'])}")
    st.success(f"Góp tháng: {fmt_vnd(res['monthly_payment_vnd'])}")
    if int(draft_pct) != committed_pct or int(draft_term) != committed_term:
        if st.button("✅ Lưu làm mặc định", use_container_width=True):
            telemetry_logger.log_correction({
                "field": "loan_percentage",
                "from": committed_pct,
                "to": int(draft_pct),
                "session_id": st.session_state.session_id,
            })
            st.session_state.user_context["loan_preference_pct"] = int(draft_pct)
            st.session_state.loan_term = int(draft_term)
            st.toast("Đã lưu sở thích mới!")
            st.rerun()


def render_charging_map(stations: list[dict[str, Any]]) -> None:
    """Render map + caption list of charging stations (NhatVi)."""
    import pandas as pd  # local import — pandas only needed here

    if not stations:
        return
    map_data = []
    for s in stations:
        name = (
            s.get("location_name") or s.get("name") or "Trạm sạc VinFast"
        )
        coords = s.get("coordinates") or s
        lat = coords.get("lat")
        lon = coords.get("lon") or coords.get("lng")
        if lat is not None and lon is not None:
            map_data.append({"name": name, "lat": float(lat), "lon": float(lon)})
    if not map_data:
        st.warning("Không có dữ liệu tọa độ để hiển thị bản đồ.")
        return
    df = pd.DataFrame(map_data)
    st.subheader("📍 Bản đồ trạm sạc")
    st.map(df)
    for s in stations:
        name = s.get("location_name") or s.get("name") or "Trạm sạc"
        addr = s.get("address") or "Đang cập nhật địa chỉ"
        types = s.get("charger_types") or []
        st.caption(f"- **{name}**: {addr} ({', '.join(types)})")


def render_sidebar() -> None:
    with st.sidebar:
        # ── Model selector (NhatVi) ──────────────────────────────────────
        st.header("⚙️ Chọn mô hình AI")
        model_labels = list(MODEL_OPTIONS.keys())
        current = st.session_state.get("model_label", _DEFAULT_MODEL)
        try:
            idx = model_labels.index(current)
        except ValueError:
            idx = 0
        chosen = st.selectbox("Model", model_labels, index=idx, key="_model_selector")
        if chosen != st.session_state.get("model_label"):
            st.session_state.model_label = chosen
            st.rerun()

        st.divider()

        # ── Live agent status (NhatVi) ───────────────────────────────────
        st.session_state.sidebar_status_area = st.empty()
        status = st.session_state.get("agent_status", "")
        if status:
            st.session_state.sidebar_status_area.info(status)
        else:
            st.session_state.sidebar_status_area.caption("⚡ Agent đang sẵn sàng.")

        st.divider()

        # ── Vehicle card + loan + booking (dev) ──────────────────────────
        vehicle = st.session_state.get("last_vehicle")
        if vehicle:
            st.markdown(
                f"<h2 style='margin-bottom:0;'>🚗 {vehicle.get('model_name')}</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-size: 1.35rem; font-weight: 700; color: #0088cc;'>"
                f"Giá: {fmt_vnd(vehicle.get('price_without_battery_vnd'))}</p>",
                unsafe_allow_html=True,
            )
            st.divider()
            render_loan_sidebar(vehicle)
            st.divider()
            if st.button("📅 Đặt lịch lái thử", use_container_width=True, type="primary"):
                st.session_state.pending_input = (
                    f"Tôi muốn đặt lịch lái thử {vehicle.get('model_id')}."
                )
                st.rerun()

        st.divider()

        # ── Memory & feedback expanders (dev) ────────────────────────────
        with st.expander("🧠 Trí nhớ AI", expanded=False):
            ctx = st.session_state.get("user_context") or {}
            if ctx:
                for k, v in ctx.items():
                    st.write(f"- **{k}**: `{v}`")
            else:
                st.caption("_(chưa học được sở thích nào)_")

        with st.expander("📊 Phản hồi", expanded=False):
            st.write(f"Đánh giá lưu: `{count_jsonl('feedback.jsonl')}`")

        if st.button("🗑️ Làm mới hội thoại", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.query_params["thread_id"] = new_id
            for key in (
                "session_id", "user_context", "chat_history", "last_vehicle",
                "loan_term", "show_loan", "pending_input", "feedback_map",
                "agent_status", "model_label",
            ):
                st.session_state.pop(key, None)
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    init_session_state()
    render_sidebar()

    st.markdown(
        """<div class="vssa-header"><h1>VSSA Trợ lý VinFast</h1></div>""",
        unsafe_allow_html=True,
    )

    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.info("Chào bạn! VinFast có thể giúp gì cho bạn hôm nay?")

        for idx, entry in enumerate(st.session_state.chat_history):
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])
                p = entry.get("payload") or {}

                # Charging stations map — NhatVi new feature
                if entry["role"] == "assistant":
                    data = p.get("charging_stations")
                    if data:
                        stations = (
                            data if isinstance(data, list)
                            else data.get("stations", [])
                        )
                        render_charging_map(stations)

                # Booking confirmation
                if p.get("cta") == "test_drive_booked":
                    sr = p.get("showroom") or {}
                    st.success(
                        f"✨ **Đã chốt lịch!** Tại {sr.get('showroom_name')} "
                        f"lúc {fmt_slot(sr.get('slot'))}."
                    )

                # Feedback buttons (dev)
                if entry["role"] == "assistant":
                    f_state = st.session_state.feedback_map.get(idx)
                    c1, c2, c3, _ = st.columns([0.17, 0.23, 0.15, 0.45])
                    with c1:
                        btn_type = "primary" if f_state == "like" else "secondary"
                        if st.button("👍 Hữu ích", key=f"like_{idx}", type=btn_type):
                            st.session_state.feedback_map[idx] = "like"
                            log_feedback("Helpful", entry["content"])
                            st.toast("Đã ghi nhận phản hồi 'Hữu ích'!")
                            st.rerun()
                    with c2:
                        btn_type = "primary" if f_state == "dislike" else "secondary"
                        if st.button("👎 Không hữu ích", key=f"dislike_{idx}", type=btn_type):
                            st.session_state.feedback_map[idx] = "dislike"
                            log_feedback("Not Helpful", entry["content"])
                            st.toast("Đã ghi nhận phản hồi 'Không hữu ích'!")
                            st.rerun()
                    with c3:
                        if st.button("📞 CSKH", key=f"cskh_{idx}", help="Liên hệ hỗ trợ"):
                            st.toast("Đang kết nối với tổng đài VinFast...")

    # Suggestion chip (shown only before first message)
    if not st.session_state.chat_history:
        if st.button(f"💡 Gợi ý: {DEFAULT_QUERY}", use_container_width=True):
            st.session_state.pending_input = DEFAULT_QUERY
            st.rerun()

    # Chat input — must be at top level (outside columns) to pin to viewport bottom
    prompt = st.chat_input("Hỏi VSSA về xe VinFast...")
    if prompt:
        st.session_state.pending_input = prompt
        st.rerun()

    # Drain pending input from chat_input OR sidebar button
    if st.session_state.pending_input:
        user_input = st.session_state.pop("pending_input")
        process_user_message(user_input, chat_container)


if __name__ == "__main__":
    main()
