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
streamlit run app/streamlit_app.py
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
