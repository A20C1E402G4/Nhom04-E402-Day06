# VinFast Smart Sales Agent (VSSA) — Prototype

VSSA là một trợ lý bán hàng thông minh (AI Agent) được thiết kế riêng cho hệ sinh thái xe điện VinFast. Dự án sử dụng kiến trúc Agentic Workflow để cung cấp trải nghiệm tư vấn cá nhân hóa, tính toán tài chính thời gian thực và tự động hóa quy trình đặt lịch lái thử.

Video demo: https://drive.google.com/file/d/16YEelN6AOqRAZFIrNTP3zIOJXgGf9znQ/view?usp=drivesdk

---

## Mô tả tính năng cốt lõi

1.  **Personal Sales Concierge:** Tư vấn chọn xe dựa trên nhu cầu, ngân sách và thói quen sử dụng của khách hàng.
2.  **Real-time Finance Planner:** Tích hợp bộ công cụ tính toán lãi suất vay, gốc phải trả hàng tháng từ các ngân hàng đối tác.
3.  **Test Drive Optimizer:** Tìm kiếm Showroom gần nhất có sẵn mẫu xe yêu cầu và hỗ trợ đặt lịch lái thử ngay lập tức.
4.  **Charging Station Locator:** Tra cứu danh sách trạm sạc VinFast theo khu vực kèm thông tin loại sạc (AC/DC).
5.  **Data-Driven Correction:** AI tự động cập nhật phương án tài chính khi người dùng điều chỉnh thông số trên giao diện Streamlit.

---

## Tools

-   **Core Engine:** LangGraph (Stateful Multi-Turn Agent).
-   **LLM Support:** OpenAI GPT-4o (mặc định), Gemini 1.5 Flash, hoặc Local LLM (qua Ollama/Qwen 2.5).
-   **Frontend:** Streamlit với hệ thống tương tác Visual Components.
-   **Database:** JSON Local (Mock data) & SQLite (Lưu trữ trạng thái hội thoại/Memory).
-   **Telemetry:** Hệ thống logging ghi lại Lead Generation và Failed Intents.

---

## 📦 Hướng dẫn cài đặt

### 1. Cài đặt môi trường
Yêu cầu Python 3.10+. Khuyến khích sử dụng `uv` hoặc `venv`.

```bash
# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Hoặc .venv\Scripts\activate trên Windows

# Cài đặt phụ thuộc
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường
Tạo file `.env` tại thư mục gốc và cấu hình API Key:

```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
LLM_PROVIDER=openai  # Hoặc 'gemini', 'local'
```

### 3. Chạy ứng dụng
```bash
streamlit run app/streamlit_app.py
```

---

## 📂 Cấu trúc thư mục

-   `app/`: Chứa giao diện Streamlit và logic xử lý UI.
-   `src/vssa_agent/`: Logic cốt lõi của Agent.
    -   `graph.py`: Định nghĩa LangGraph Workflow.
    -   `tools.py`: Các công cụ thực thi (Finance, Showroom, Vehicles).
    -   `prompts/`: System prompt định hình persona của Sales Expert.
-   `data/`: Database dưới dạng JSON (Vehicles, Showrooms, Finance, Charging Stations).
-   `telemetry/`: Hệ thống theo dõi hiệu năng và tỷ lệ chuyển đổi Lead.

---

## Luồng hoạt động (Agentic Path)

1.  **Happy Path:** Khách hỏi -> Agent gọi Tool -> Trả kết quả kèm link showroom.
2.  **Correction Path:** Khách chỉnh Slider % vay trên UI -> Agent nhận diện `user_context` thay đổi -> Query lại tool tài chính và cập nhật UI.
3.  **Fallback Path:** Khách hỏi ngoài phạm vi -> Agent xin lỗi và gợi ý liên hệ Sales thật để tránh ảo giác.

---

---

## Phân công
| Thành viên | Phần | Output |
|-----------|------|--------|
| 2A202600064 Hoàng Đinh Duy Anh | Canvas + failure modes + prompt engineering + tool calling + data structure | spec/spec-final.md phần 1, 4 |
| 2A202600497 Trần Nhật Vĩ | UI propotype + User stories 4 paths + prompt engineering + Canvas | spec/spec-final.md phần 1, 2, prototype/prompt-tests.md |
| 2A202600486 Nguyễn Tiến Huy Hoàng | Eval metrics + threshold + UI propotype + demo scripts + test case | spec/spec-final.md phần 3, demo/slides.pdf |
| 2A202600312 Trần Thanh Phong | Eval metric + Tool call + ROI + demo slide + prompt engineering | spec/spec-final.md phần 3, 5, demo/demo-script.md, demo/slides.pdf |

---




