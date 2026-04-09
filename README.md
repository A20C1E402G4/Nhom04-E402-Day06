# VSSA — VinFast Smart Sales Agent

> **AI Product Hackathon · Track: VinFast · Nhóm 04**
>
> Khách hàng mua xe điện thường bị "ngợp" bởi thông số kỹ thuật, bài toán chi phí thuê pin và các gói vay.
> VSSA là Agent thông minh giúp cá nhân hoá tư vấn 24/7, tự động lập phương án tài chính và chốt lịch lái thử trong 1
> phút.

---

## Demo — Case 1: Personal Sales Concierge

The primary flow: a customer states their budget and family size, the agent recommends a vehicle, calculates a
financing plan, and books a test drive — all within a single chat session.

```
User  → "Tôi có 800 triệu, nhà 4 người, ở chung cư, nên mua xe nào?"
Agent → [calls get_all_vehicles → get_vehicle_data → calculate_loan]
      → "VF 7 base phù hợp nhất. Trả trước 255tr, góp ~8.5tr/tháng."
      → [renders vehicle card + loan table in sidebar]

User  → "Cho tôi đặt lịch lái thử"
Agent → "Bạn đang ở khu vực nào?" → [calls find_showrooms]
      → "VinFast Thảo Điền, 9:00 sáng mai. Cho mình tên và SĐT?"
      → [calls book_test_drive] → confirms booking
```

---

## Table of Contents

1. [Product Canvas](#1-product-canvas)
2. [Architecture](#2-architecture)
3. [Getting Started](#3-getting-started)
4. [Agent Tools](#4-agent-tools)
5. [Data](#5-data)
6. [UI & UX Paths](#6-ui--ux-paths)
7. [Learning Signal & Data Flywheel](#7-learning-signal--data-flywheel)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Failure Modes & Mitigations](#9-failure-modes--mitigations)
10. [Testing](#10-testing)
11. [Project Structure](#11-project-structure)
12. [Team](#12-team)

---

## 1. Product Canvas

|              | Value                                                                                | Trust                                                                          | Feasibility                                                                  |
|--------------|--------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **Question** | Who is the user? What is the pain? What does AI solve?                               | What if AI is wrong? How does the user correct it?                             | Cost/latency? Main risk?                                                     |
| **Answer**   | First-time EV buyers afraid of complex calculations. AI acts as a 24/7 Sales Expert. | AI may quote wrong price/promo. Always shows disclaimer + link to source data. | ~$0.2/session. Latency <5s. Risk: hallucination on specs not yet in dataset. |

**Automation vs. Augmentation:** Augmentation — AI is the funnel that guides the customer. Contract signing and
vehicle handover still require a human for legal and experiential reasons.

**ROI Scenarios:**

|            | Conservative             | Realistic                      | Optimistic                             |
|------------|--------------------------|--------------------------------|----------------------------------------|
| Volume     | 1,000 customers/month    | 5,000 customers/month          | 20,000 customers/month                 |
| Cost       | $200                     | $800                           | $2,500                                 |
| Output     | 50 leads                 | 500 leads                      | 3,000 leads + 200 test drives          |
| Equivalent | Cheaper than 1 sales rep | Matches a 10-person sales team | Becomes a primary global sales channel |

**Kill criteria:** Stop if cost-per-lead from AI exceeds traditional advertising for 3 consecutive months.

---

## 2. Architecture

```
┌─────────────────────────────────────────┐
│           Streamlit UI                  │
│  Single-column chat · Fixed input bar   │
│  Sidebar: model selector, vehicle card, │
│  loan panel, memory, feedback           │
│  Streaming status per LangGraph node    │
└──────────────────┬──────────────────────┘
                   │  invoke_agent() generator
                   ▼
┌─────────────────────────────────────────┐
│         LangGraph StateGraph            │
│                                         │
│  agent_node ──► tools_condition         │
│      ▲                │                 │
│      └──── ToolNode ◄─┘                 │
│                                         │
│  State:  messages + user_context        │
│          + tool_turns (max 3)           │
│  Memory: SqliteSaver → vssa_state.sqlite│
└──────────┬───────────────┬──────────────┘
           │               │
    ┌──────┴──────┐  ┌──────┴────────────────┐
    │  LLM Layer  │  │  7 Tools              │
    │             │  │                       │
    │ OpenAI      │  │ get_all_vehicles      │
    │ GPT-4o      │  │ get_vehicle_data      │
    │   OR        │  │ calculate_loan        │
    │ Ollama      │  │ get_promotion         │
    │ Qwen 2.5 /  │  │ find_showrooms        │
    │ Llama 3.2   │  │ find_charging_stations│
    └─────────────┘  │ book_test_drive       │
                     └───────────────────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
           data/*.json                logs/*.jsonl
           (mock data)                (flywheel)
```

### Agent State

```python
class UserContext(TypedDict, total=False):
    budget_vnd: int
    family_size: int
    housing: str  # "apartment" | "house"
    loan_preference_pct: int
    preferred_model: str
    location_hint: str


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_context: UserContext
    tool_turns: int  # hard stop at 3
```

---

## 3. Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key **OR** [Ollama](https://ollama.com) running locally

### Installation

```bash
git clone https://github.com/A20C1E402G4/vssa.git
cd vssa

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### Run

```bash
uv run streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### Local Model (Optional)

```bash
# Install Ollama from https://ollama.com
ollama pull qwen2.5
# Switch to "Local — Qwen 2.5" in the sidebar model selector
```

### Run Tests

```bash
uv run pytest -q
# Expected: 17 passed
```

> **Note:** After modifying any `data/*.json` file, restart the Streamlit process to clear `lru_cache`.

---

## 4. Agent Tools

All tools use **Pydantic `args_schema`** — the LLM cannot pass values outside the defined schema (e.g. unknown
model IDs, loan percentages > 95, invalid slot formats). This is the primary hallucination guard.

| Tool                     | Trigger                                    | Key Constraint                                                             |
|--------------------------|--------------------------------------------|----------------------------------------------------------------------------|
| `get_all_vehicles`       | First turn of any vehicle inquiry          | No args — always called before recommending                                |
| `get_vehicle_data`       | Specific model detail needed               | `model_id: Literal["VF3","VF5","VF6","VF7","VF8","VF9","VF_Wild"]`         |
| `calculate_loan`         | User mentions "trả góp", "vay", "lãi suất" | `loan_percentage ∈ [10,95]`, `duration_months ∈ {12,24,36,48,60,72,84,96}` |
| `get_promotion`          | Promotion check for a model                | `model_id: VehicleModelId`                                                 |
| `find_showrooms`         | User wants to book a test drive            | `model_id`, optional `district`                                            |
| `find_charging_stations` | User asks about charging                   | optional `query: str` (text search on location/address)                    |
| `book_test_drive`        | All booking info collected                 | `phone ≥ 8 digits`, `slot` must be ISO datetime from `find_showrooms`      |

### System Prompt Key Rules

1. **Call `get_all_vehicles` first** before any recommendation — never guess price from training data.
2. Only call `calculate_loan` when the user explicitly mentions financing keywords — never auto-show loan details.
3. Booking uses a **5-step protocol**: ask location → `find_showrooms` → confirm slot → collect name/phone →
   `book_test_drive`.
4. Charging stations: ask area first → `find_charging_stations` → render map in chat.
5. Max 3 tool turns per user message — then ask for clarification instead of looping.
6. Off-topic questions (not VinFast related): politely refuse and redirect.

### In-Session Learning Signal

Every agent turn receives a dynamic `<user_profile>` SystemMessage built from `AgentState.user_context`:

```
<user_profile>
- Ngân sách: 850,000,000 VND
- Gia đình: 4 người
- Tỷ lệ vay ưa thích: 60%
- Khu vực: Thủ Đức
</user_profile>
```

If a user adjusts their loan preference mid-conversation, the agent applies it automatically on the next turn.

---

## 5. Data

All data is mocked as JSON files in `data/`. No external API calls are made during the demo.

### Vehicles — 11 models

| Model ID                                     | Description     | Base price (no battery) |
|----------------------------------------------|-----------------|-------------------------|
| VF3                                          | City micro EV   | —                       |
| VF6_Base / VF6_Plus                          | Compact SUV     | —                       |
| VFe34                                        | Sedan EV        | —                       |
| VF7                                          | C-SUV base      | 850,000,000 VND         |
| VF7_Plus                                     | C-SUV premium   | 999,000,000 VND         |
| VF8_Plus                                     | D-SUV           | —                       |
| VF9_Eco / VF9_Plus_7Seats / VF9_Plus_Captain | Full-size SUV   | —                       |
| VF_Wild                                      | Off-road pickup | —                       |

### Showrooms — 11 locations

**Hà Nội (7):** Ocean Park (Gia Lâm), Smart City (Nam Từ Liêm), Bà Triệu (Hai Bà Trưng), Royal City (Thanh
Xuân), Long Biên, Nguyễn Chí Thanh (Đống Đa), TechnoPark (Gia Lâm)

**TP.HCM (4):** Thảo Điền `SR_Q9` (Thủ Đức), Đồng Khởi `SR_Q1` (Q1), Tân Định (Q1), Gò Vấp

### Banks

| Bank        | Annual Rate | Max LTV | Max Term  |
|-------------|-------------|---------|-----------|
| Vietcombank | 8.5%        | 80%     | 96 months |
| Techcombank | 9.0%        | 85%     | 84 months |

Active promotions: `VF7_FREE_CHARGE_1YR`, `VF8_0PCT_6MO`

### Charging Stations — 12 stations

Hà Nội (10) · Quảng Ninh (1) · Hải Phòng (1) · TP.HCM · Vũng Tàu

---

## 6. UI & UX Paths

The UI handles four agent states defined in the UX spec:

### Happy Path

Agent is confident and correct. UI renders:

- Vehicle card with key specs
- Financing table (only if user asked about loans)
- "Đặt lịch lái thử" button that opens the booking conversation

### Low-Confidence Path

Tool returns no result or out-of-scope question:
> "Thông tin này tôi chưa rõ. Hãy trao đổi với saler để được tư vấn kỹ hơn."

CSKH button connects to a real consultant.

### Failure Path

Agent may have quoted wrong data. Every financial figure includes:
> *Lưu ý: Các con số tính toán và giá trên đây chỉ mang tính tham khảo. Vui lòng xác nhận với Showroom.*

### Correction Path

User adjusts the loan plan in the sidebar → clicks **Xác nhận** → agent recalculates and logs the correction to
`logs/corrections.jsonl` for the data flywheel.

### Model Selector

```
☁️ OpenAI — GPT-4o       (default, requires OPENAI_API_KEY)
🏠 Local — Qwen 2.5      (requires Ollama running locally)
🏠 Local — Llama 3.2     (requires Ollama running locally)
```

---

## 7. Learning Signal & Data Flywheel

### Files Written (append-only JSONL)

| File                        | Written when                              | What it captures                                          |
|-----------------------------|-------------------------------------------|-----------------------------------------------------------|
| `logs/agent.jsonl`          | Every LangGraph node execution            | Message trace, tool calls, node name, timestamp           |
| `logs/bookings.jsonl`       | Successful `book_test_drive`              | Name, phone, model, showroom, slot → lead conversion      |
| `logs/corrections.jsonl`    | User edits loan panel → clicks Xác nhận   | Original vs. corrected loan% and term → preference signal |
| `logs/failed_intents.jsonl` | Agent hits 3-turn limit without resolving | Last user message, partial tool trace → coverage gaps     |
| `logs/metrics.json`         | On demand                                 | Tool success rate, lead-gen recall                        |

### Database (`vssa_state.sqlite`)

`SqliteSaver` (LangGraph) stores the full `AgentState` keyed by `thread_id`.
`thread_id` is a UUID written to `?thread_id=` in the browser URL — the conversation survives page refresh and
can be resumed across browser sessions.

### Flywheel Loop

```
Chat interaction → logs/*.jsonl (raw signal)
                                    │
              ┌─────────────────────┤
              ▼                     ▼
    Prompt refinement          Fine-tuning / DPO
    (system_prompt.txt)        (future — not yet wired)
              │
              ▼
    <user_profile> injected every turn
    (in-session loop — active now)
```

The in-session loop is live. The offline training loop (RLHF/DPO from `corrections.jsonl`) is the next phase.

---

## 8. Evaluation Metrics

**Precision** — for pricing and financial data (must be 100% accurate):

| Metric                 | Threshold | Red flag           |
|------------------------|-----------|--------------------|
| Price & spec accuracy  | 100%      | Any error on price |
| Tool call success rate | > 95%     | < 80%              |
| Lead gen rate          | ≥ 15%     | < 5%               |

**Recall** — for sales conversion (cast a wide net):

| Metric                     | Threshold | Red flag                            |
|----------------------------|-----------|-------------------------------------|
| Lead generation recall     | ≥ 90%     | < 70% (missed buyers)               |
| Response rate              | 100%      | < 95%                               |
| Engagement (turns/session) | > 5       | < 2 (user left after first message) |
| Severe spec errors         | < 2%      | > 5%                                |

---

## 9. Failure Modes & Mitigations

| # | Failure                        | Consequence                                     | Mitigation                                                       |
|---|--------------------------------|-------------------------------------------------|------------------------------------------------------------------|
| 1 | API outdated / down            | Agent has no data to answer                     | Fallback response: apologise + auto-notify real sales rep        |
| 2 | LLM hallucination in tool args | Agent invents a loan rate (e.g. 50%)            | Pydantic `args_schema` validates all inputs before tool executes |
| 3 | Logic loop                     | Agent calls tools repeatedly without resolution | `tool_turns` hard limit of 3; then ask user for clarification    |

---

## 10. Testing

```bash
uv run pytest -q   # 17 passed
```

### Coverage

| File                        | Tests                                                                                                                                                                                           |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `tests/test_tools.py`       | Schema rejection (bad loan%, duration, model ID); `calculate_loan` baseline VF7 @ 70%/96mo → ~8.5M/mo; showroom lookup; promo codes; booking validation + happy path with monkeypatched log dir |
| `tests/test_agent_graph.py` | `ScriptedChatModel` fake LLM; asserts tool call sequence `["get_vehicle_data", "calculate_loan"]`; booking path calls `find_showrooms`                                                          |

---

## 11. Project Structure

```
vssa/
├── app/
│   └── streamlit_app.py          # UI — 577 lines
├── data/
│   ├── vehicles.json             # 11 VinFast models
│   ├── showrooms.json            # 11 showrooms (HN + HCM)
│   ├── finance.json              # 2 banks, 2 promotions
│   └── charging_stations.json   # 12 stations
├── src/
│   ├── core/
│   │   ├── llm_provider.py       # Abstract LLMProvider base
│   │   ├── openai_provider.py    # ChatOpenAI (GPT-4o)
│   │   └── local_provider.py    # ChatOllama (Qwen 2.5 / Llama 3.2)
│   ├── telemetry/
│   │   ├── logger.py            # JSONL writers for all 4 log files
│   │   └── metrics.py           # In-memory aggregates
│   └── vssa_agent/
│       ├── config.py            # Paths, env vars, lru_cache JSON loaders
│       ├── state.py             # AgentState + UserContext TypedDicts
│       ├── schemas.py           # Pydantic args_schema per tool
│       ├── tools.py             # 7 tool implementations
│       ├── graph.py             # LangGraph StateGraph
│       └── prompts/
│           └── system_prompt.txt
├── tests/
│   ├── test_tools.py            # 15 unit tests
│   └── test_agent_graph.py      # 2 integration tests
├── logs/                        # Auto-created, gitignored
│   ├── agent.jsonl
│   ├── bookings.jsonl
│   ├── corrections.jsonl
│   └── failed_intents.jsonl
├── vssa_state.sqlite             # LangGraph cross-session checkpoints
├── DEVLOG.md                    # Full development log + personal contribution
└── pyproject.toml
```

---

## 12. Team

| Member                | Role                           | Scope                                                                                                                      |
|-----------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Trần Nhật Vĩ          | AI Research & Data             | Parts 1–3: Product canvas, user stories, data expansion (vehicles, charging stations), local LLM provider, streaming agent |
| Trần Thanh Phong      | AI Engineer                    | Parts 4–6: Tools, graph, telemetry                                                                                         |
| Nguyễn Tiến Huy Hoàng | Frontend Engineer              | Parts 1–3: UI overhaul, feedback UX, light-mode CSS                                                                        |
| Hoàng Đinh Duy Anh    | AI Engineer & Integration Lead | Parts 4–6: Agent backend scaffolding, Streamlit integration, branch merges, testing, data integrity                        |

---
