# VinFast Smart Sales Agent (VSSA) 🚗⚡

VSSA là trợ lý bán hàng AI thế hệ mới, được tối ưu hóa riêng cho hệ sinh thái xe điện VinFast. Sử dụng kiến trúc **LangGraph (Stateful Agentic Workflow)**, VSSA không chỉ trả lời câu hỏi mà còn chủ động dẫn dắt khách hàng từ khâu chọn xe, lập phương án tài chính đến chốt lịch lái thử trong 1 phút.

---

## 🌟 Tính năng đột phá

*   **Tư vấn cá nhân hóa (Personal Sales Expert):** Đề xuất dòng xe (VF 3, VF 5, VF 7,...) dựa trên ngân sách, thói quen di chuyển và số lượng thành viên gia đình.
*   **Chuyên gia tài chính (Smart Finance Planner):** Tự động tính toán tiền trả trước, gốc + lãi hàng tháng theo từng gói vay (80%, 24 tháng, 8 năm) và chính sách thuê pin/mua pin.
*   **Tối ưu lịch lái thử (Test Drive Optimizer):** Tìm kiếm showroom gần nhất qua Google Maps API, kiểm tra tình trạng xe demo và xác nhận lịch hẹn tức thì.
*   **Mạng lưới trạm sạc (Power Station Hub):** Tra cứu nhanh trạm sạc AC/DC quanh khu vực khách sống để giải quyết triệt để nỗi lo về hạ tầng.
*   **Giao diện tương tác Live (Reactive UI):** Tự động cập nhật số liệu tư vấn khi người dùng điều chỉnh thông số trên thanh kéo (Slider) hoặc nút chọn.

---

## 🛠 Công nghệ sử dụng

| Thành phần | Công nghệ |
| :--- | :--- |
| **Logic Core** | LangGraph (Stateful Multi-turn Agent) |
| **Mô hình ngôn ngữ** | GPT-4o / Gemini 1.5 Flash / Qwen 2.5 (Local) |
| **Giao diện** | Streamlit (Python-based Web App) |
| **Lưu trữ** | SQLite (Conversation Memory) & JSON (Product Data) |
| **Theo dõi** | Lead Generation Telemetry & JSONL Logging |

---

## 🚀 Hướng dẫn cài đặt & Thực thi

### 1. Chuẩn bị môi trường
Yêu cầu Python 3.10+. Khuyến khích dùng `venv`.

```bash
# Clone dự án
git clone <repository_url>
cd Nhom04-E402-Day06

# Tạo và kích hoạt môi trường ảo
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Cấu hình API Key
Tạo file `.env` tại thư mục gốc từ mẫu `.env.example`:

```env
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
LLM_PROVIDER=gemini  # Tùy chọn: openai, gemini, hoặc local
```

### 3. Chạy ứng dụng
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

## 📂 Kiến trúc dự án

```text
├── app/                  # Streamlit UI & Event Handlers
├── data/                 # Catalog: Vehicles, Showrooms, Charging Stations
├── src/vssa_agent/       # Brain of the system
│   ├── graph.py          # LangGraph State Machine
│   ├── tools.py          # Function Calling (Finance, Bookings, Search)
│   └── prompts/          # System Prompts & Sales Persona
├── telemetry/            # Logs for Lead Gen & Conversation Analysis
└── tests/                # System Evaluation & Unit Tests
```

---

## 🛡 Chiến lược an toàn & Chính xác

1.  **Strict Accuracy:** Ưu tiên 100% chính xác về thông số giá và khuyến mãi thông qua việc truy vấn trực tiếp DB nội bộ.
2.  **Smart Fallback:** Tự động phát hiện ý định ngoài phạm vi và gợi ý kết nối với nhân viên tư vấn thật.
3.  **No Hallucinations:** Ràng buộc kết quả đầu ra bằng logic validation trước khi hiển thị cho người dùng.

---
**Nhóm:** VinSales AI-Powered | **Sản phẩm tham dự AI Product Hackathon**
