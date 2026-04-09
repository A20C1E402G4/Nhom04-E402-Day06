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
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from core.openai_provider import OpenAIProvider  # noqa: E402
from telemetry import logger as telemetry_logger  # noqa: E402
from telemetry import metrics as telemetry_metrics  # noqa: E402
from vssa_agent.config import CHECKPOINT_DB_PATH, LOGS_DIR  # noqa: E402
from vssa_agent.graph import build_graph  # noqa: E402
from vssa_agent.tools import _calculate_loan_impl  # noqa: E402

st.set_page_config(page_title="VSSA — VinFast Smart Sales Agent", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Styling (Premium ChatGPT Aesthetics)
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
        max-width: 900px;
    }
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }
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

    /* Feedback Buttons Styling */
    .stButton button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        color: var(--text-color) !important;
        opacity: 0.6;
        font-size: 0.8rem !important;
        padding: 2px 10px !important;
        border-radius: 8px !important;
        height: auto !important;
        white-space: nowrap !important; /* Ngăn nhảy dòng chữ */
    }
    .stButton button[kind="secondary"]:hover {
        opacity: 1;
        border-color: #0088cc !important;
        color: #0088cc !important;
    }

    /* Suggestions Styling */
    div[data-testid="stVerticalBlock"] > div:nth-last-child(2) .stButton button {
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        font-size: 0.85rem !important;
        border-radius: 12px !important;
    }

    /* st.status tweaks */
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

DEFAULT_QUERY = "Tôi có 800 triệu, nhà 4 người, ở chung cư, nên mua xe nào?"
OTHER_QUERY = "Các dòng xe SUV của VinFast có gì nổi bật?"
JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
LOAN_KEYWORDS = ("trả góp", "vay", "tài chính", "lãi suất", "trả trước", "góp", "khoản vay")
TERM_OPTIONS = [24, 36, 48, 60, 72, 84, 96]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_vnd(value: int | float | None) -> str:
    if value is None: return "—"
    return f"{int(value):,}".replace(",", ".") + " ₫"

def fmt_slot(iso: str) -> str:
    try: return datetime.fromisoformat(iso).strftime("%d/%m/%Y %H:%M")
    except: return iso

def parse_agent_payload(text: str) -> dict[str, Any] | None:
    match = JSON_BLOCK_RE.search(text or "")
    if not match: return None
    try: return json.loads(match.group(1))
    except: return None

def strip_json_block(text: str) -> str:
    return JSON_BLOCK_RE.sub("", text or "").strip()

def count_jsonl(filename: str) -> int:
    path = LOGS_DIR / filename
    if not path.exists(): return 0
    with path.open("r", encoding="utf-8") as f: return sum(1 for _ in f)

def log_feedback(sentiment: str, content: str):
    log_path = LOGS_DIR / "feedback.jsonl"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": st.session_state.get("session_id", "unknown"),
        "sentiment": sentiment,
        "content": content[:200] + "..." if len(content) > 200 else content
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_compiled_graph():
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    saver = SqliteSaver(conn)
    return build_graph(OpenAIProvider().chat_model(), checkpointer=saver)

def get_or_create_thread_id() -> str:
    if "thread_id" in st.query_params: return st.query_params["thread_id"]
    tid = str(uuid.uuid4()); st.query_params["thread_id"] = tid; return tid

def load_persisted_user_context(thread_id: str) -> dict[str, Any]:
    try:
        app = get_compiled_graph()
        snapshot = app.get_state({"configurable": {"thread_id": thread_id}})
        return dict((snapshot.values or {}).get("user_context") or {})
    except: return {}

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
    st.session_state.setdefault("is_processing", False)
    st.session_state.setdefault("pending_input", None)

# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.image("https://vinfastauto.com/themes/custom/vinfast_theme/logo.png", width=140)
        st.title("VSSA Control")
        st.divider()
        vehicle = st.session_state.get("last_vehicle")
        if vehicle:
            st.markdown(f"<h2 style='margin-bottom:0;'>🚗 {vehicle.get('model_name')}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 1.35rem; font-weight: 700; color: #ff3b30;'>Giá: {fmt_vnd(vehicle.get('price_without_battery_vnd'))}</p>", unsafe_allow_html=True)
            st.divider()
            render_loan_sidebar(vehicle)
            st.divider()
            if st.button("📅 Đặt lịch lái thử", use_container_width=True, type="primary"):
                st.session_state.pending_input = f"Tôi muốn đặt lịch lái thử {vehicle.get('model_id')}."
                st.rerun()
        st.divider()
        with st.expander("🧠 Trí nhớ AI", expanded=False):
            ctx = st.session_state.get("user_context") or {}
            for k, v in ctx.items(): st.write(f"- **{k}**: `{v}`")
        
        with st.expander("📊 Phản hồi", expanded=False):
            st.write(f"Đánh giá lưu: `{count_jsonl('feedback.jsonl')}`")

        if st.button("🗑️ Làm mới hội thoại", use_container_width=True):
            new_id = str(uuid.uuid4()); st.query_params["thread_id"] = new_id
            for key in ("session_id", "user_context", "chat_history", "last_vehicle", "loan_term", "is_processing", "pending_input"):
                st.session_state.pop(key, None)
            st.rerun()

def render_loan_sidebar(vehicle: dict[str, Any]) -> None:
    st.markdown("**💰 Dự tính trả góp**")
    price = int(vehicle.get("price_without_battery_vnd") or 0)
    if price <= 0: return
    committed_pct = int(st.session_state.user_context.get("loan_preference_pct", 70))
    committed_term = int(st.session_state.loan_term)
    draft_pct = st.slider("Tỷ lệ vay (%)", 0, 80, value=committed_pct, step=5)
    try: t_idx = TERM_OPTIONS.index(committed_term)
    except: t_idx = len(TERM_OPTIONS) - 1
    draft_term = st.selectbox("Kỳ hạn (tháng)", TERM_OPTIONS, index=t_idx)
    res = _calculate_loan_impl(price, int(draft_pct), int(draft_term))
    st.info(f"Trả trước: {fmt_vnd(res['down_payment_vnd'])}")
    st.success(f"Góp tháng: {fmt_vnd(res['monthly_payment_vnd'])}")
    if int(draft_pct) != committed_pct or int(draft_term) != committed_term:
        if st.button("✅ Lưu làm mặc định", use_container_width=True):
            telemetry_logger.log_correction({"field": "loan_percentage", "from": committed_pct, "to": int(draft_pct), "session_id": st.session_state.session_id})
            st.session_state.user_context["loan_preference_pct"] = int(draft_pct); st.session_state.loan_term = int(draft_term)
            st.toast("Đã lưu sở thích mới!"); st.rerun()

# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main() -> None:
    init_session_state()
    render_sidebar()

    st.markdown("""<div class="vssa-header"><h1>VSSA Trợ lý VinFast</h1></div>""", unsafe_allow_html=True)

    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.info("Chào bạn! VinFast có thể giúp gì cho bạn hôm nay?")
        
        for idx, entry in enumerate(st.session_state.chat_history):
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])
                p = entry.get("payload") or {}
                if p.get("cta") == "test_drive_booked":
                    sr = p.get("showroom") or {}
                    st.success(f"✨ **Đã chốt lịch!** Tại {sr.get('showroom_name')} lúc {fmt_slot(sr.get('slot'))}.")
                
                # Feedback buttons for assistant
                if entry["role"] == "assistant":
                    c1, c2, c3, _ = st.columns([0.17, 0.23, 0.15, 0.45])
                    with c1:
                        if st.button("👍 Hữu ích", key=f"like_{idx}"):
                            log_feedback("Helpful", entry["content"])
                            st.toast("Cảm ơn bạn đã phản hồi!")
                    with c2:
                        if st.button("👎 Không hữu ích", key=f"dislike_{idx}"):
                            log_feedback("Not Helpful", entry["content"])
                            st.toast("Chúng mình sẽ cải thiện hơn!")
                    with c3:
                        if st.button("📞 CSKH", key=f"cskh_{idx}", help="Liên hệ hỗ trợ"):
                            st.toast("Đang kết nối với tổng đài VinFast...")

    # Processing block
    if st.session_state.is_processing:
        user_msg = st.session_state.chat_history[-1]["content"]
        with st.chat_message("assistant"):
            with st.status("VSSA đang kết nối...", expanded=True) as status:
                try:
                    app = get_compiled_graph()
                    config = {"configurable": {"thread_id": st.session_state.session_id}}
                    state_input = {"messages": [HumanMessage(content=user_msg)], "user_context": st.session_state.user_context, "tool_turns": 0}
                    
                    final_reply = ""
                    for event in app.stream(state_input, config=config):
                        for node_name, output in event.items():
                            if node_name == "agent":
                                status.update(label="VSSA đang suy nghĩ...", state="running")
                                if "messages" in output:
                                    last_msg = output["messages"][-1]
                                    final_reply = getattr(last_msg, "content", "")
                            elif node_name == "tools":
                                status.update(label="VSSA đang xử lý dữ liệu (Xe/Trả góp)...", state="running")
                    
                    status.update(label="Xử lý hoàn tất!", state="complete", expanded=False)
                    
                    cleaned = strip_json_block(final_reply)
                    payload = parse_agent_payload(final_reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": cleaned, "payload": payload})
                    
                    if payload:
                        if payload.get("vehicle"): st.session_state.last_vehicle = payload["vehicle"]
                        if "loan" in payload:
                            l = payload["loan"]; st.session_state.loan_term = int(l.get("duration_months", 96))
                
                except Exception as e:
                    status.update(label="Gặp lỗi khi xử lý", state="error")
                    st.error(f"⚠️ Lỗi: {e}")
            
            st.session_state.is_processing = False
            st.rerun()

    # Suggestions row above chat input
    if not st.session_state.chat_history:
        if st.button(f"💡 Gợi ý: {DEFAULT_QUERY}", use_container_width=True):
            st.session_state.pending_input = DEFAULT_QUERY; st.rerun()

    prompt = st.chat_input("Hỏi VSSA về xe VinFast...")
    if prompt:
        st.session_state.pending_input = prompt; st.rerun()

    if st.session_state.pending_input:
        user_input = st.session_state.pop("pending_input")
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.is_processing = True; st.rerun()

if __name__ == "__main__":
    main()
