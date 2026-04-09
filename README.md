# VinFast Smart Sales Agent (VSSA) 🚗⚡

VSSA là giải pháp AI Agent cao cấp hỗ trợ bán hàng và tư vấn xe điện VinFast. Dự án sử dụng **LangGraph** để xây dựng luồng tư vấn có trạng thái (stateful), kết hợp hệ thống **Telemetry** thời gian thực để tối ưu quy trình chuyển đổi khách hàng (Lead Generation).

---

## 🌟 Tính năng cốt lõi

*   **Tư vấn cá nhân hóa:** Phân tích nhu cầu để đề xuất trong 11 mẫu xe VinFast từ database (`vehicles.json`).
*   **Lập kế hoạch tài chính:** Tích hợp 2 ngân hàng đối tác và 2 chương trình ưu đãi (`finance.json`), tính toán lãi suất và gốc hàng tháng chính xác.
*   **Tự động hóa lịch lái thử:** Tra cứu 11 showroom tại Hà Nội & TP.HCM, xác nhận lịch hẹn và lưu vết booking.
*   **Tìm kiếm trạm sạc:** Định vị 12 trạm sạc trọng điểm tại các khu vực HN, QN, HP, HCM, VT.
*   **Hỗ trợ đa mô hình:** Chạy linh hoạt trên GPT-4o, Gemini hoặc Local LLM (Qwen 2.5 qua Ollama).

---

## 🛠 Kiến trúc hệ thống

```text
vssa/
├── app/
│   └── streamlit_app.py        # UI trung tâm, quản lý tương tác người dùng
├── data/                       # CSDL Mock: Xe, Showroom, Tài chính, Trạm sạc
├── src/
│   ├── core/                   # LLM Provider abstraction (OpenAI, Gemini, Local)
│   ├── telemetry/              # Logging JSONL & phân tích chỉ số chuyển đổi
│   └── vssa_agent/             # Brain: LangGraph, Tools (7 tools), Pydantic Schemas
├── tests/                      # 17 unit tests cho Tools & Integration tests cho Graph
├── logs/                       # Nhật ký Agent, Bookings, Corrections, Failed Intents
└── vssa_state.sqlite           # Bộ nhớ dài hạn (Cross-session memory)
```

---

## 🚀 Cài đặt & Sử dụng

### 1. Khởi tạo môi trường
```bash
# Tạo môi trường ảo
python -m venv .venv
.\.venv\Scripts\activate  # Windows

# Cài đặt thư viện (LangGraph, Streamlit, Pydantic, etc.)
pip install -r requirements.txt
```

### 2. Cấu hình
Tạo file `.env` tại thư mục gốc:
```env
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
LLM_PROVIDER=openai  # Hoặc 'gemini', 'local'
```

### 3. Thực thi
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Telemetry & Chất lượng dịch vụ

Hệ thống lưu trữ nhật ký tại thư mục `logs/` để phục vụ Data Flywheel:
*   `agent.jsonl`: Theo dõi độ trễ và token usage.
*   `bookings.jsonl`: Ghi nhận tỷ lệ chốt lịch lái thử thành công.
*   `corrections.jsonl`: Theo dõi hành vi sửa đổi tham số tài chính của khách hàng.
*   `failed_intents.jsonl`: Thu thập các yêu cầu chưa được đáp ứng để mở rộng API.

**Cơ chế Checkpoint:** Sử dụng SQLite để duy trì ngữ cảnh hội thoại ngay cả khi người dùng tải lại trang, đảm bảo trải nghiệm tư vấn liền mạch.

---
**Nhóm:** VinSales AI-Powered | **AI Product Hackathon Project**
