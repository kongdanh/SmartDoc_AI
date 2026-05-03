<div align="center">

# 🧠 SmartDoc AI

Hệ thống khai thác tri thức từ tài liệu — kết hợp **GraphRAG** và **Standard RAG**, cho phép so sánh song song hai phương pháp.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## Tổng quan

SmartDoc AI cho phép upload tài liệu (PDF, DOCX, TXT), tự động xây dựng index và trả lời câu hỏi dựa trên nội dung tài liệu bằng hai phương pháp:

- **GraphRAG** (Microsoft) — Trích xuất entities, relationships, community reports → trả lời câu hỏi tổng hợp, phức tạp
- **Standard RAG** (ChromaDB) — Vector similarity search với local embeddings → trả lời câu hỏi cụ thể, nhanh

## Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Streamlit-AGraph |
| GraphRAG | Microsoft GraphRAG 3.0, LanceDB, Parquet, LiteLLM |
| Standard RAG | ChromaDB, LangChain, HuggingFace Embeddings |
| LLM | OpenRouter (miễn phí) |
| Embedding | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Xử lý tài liệu | PyMuPDF, python-docx |

## Tính năng

- **Chat AI** — Streaming real-time (SSE), multi-turn, lưu lịch sử
- **Query GraphRAG** — Local / Global / DRIFT / Direct search
- **Compare RAG** — Chạy song song cả hai phương pháp, hiển thị so sánh
- **Knowledge Graph** — Trực quan hóa đồ thị tri thức
- **Upload & Quản lý** — Đa domain, auto-index, xóa file/domain
- **Tối ưu hiệu năng** — Parallel embedding (4 workers), smart cache (24h TTL), exponential backoff retry

## Cài đặt

```bash
# 1. Clone & setup
git clone <repository-url>
cd code
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# 2. Cấu hình .env
```

Tạo file `.env`:

```env
LLM_PROVIDER=openrouter
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
LLM_API_KEY=your_key_here
LLM_RPM=5

EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

DATA_DIR=./data
INDEX_DIR=./indexes
SERVER_PORT=8001
```

> Đăng ký API key miễn phí tại [openrouter.ai](https://openrouter.ai)

## Khởi chạy

```bash
# Terminal 1 — Backend
python main.py

# Terminal 2 — Frontend
streamlit run app_ui.py
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| API Docs | http://localhost:8001/docs |

## Sử dụng

1. **Overview** → Nhập tên domain → Upload files → Hệ thống auto-index Standard RAG
2. **Reindex** → Chạy GraphRAG indexing (tạo Knowledge Graph)
3. **Chat** → Chọn domain → Đặt câu hỏi
4. **Query** → Truy vấn nâng cao qua GraphRAG (Local/Global/DRIFT)
5. **Compare RAG** → So sánh kết quả giữa hai phương pháp
6. **Knowledge Graph** → Xem trực quan đồ thị tri thức

## Cấu trúc project

```
├── main.py                  # FastAPI server
├── app_ui.py                # Streamlit entry point
├── standard_rag.py          # Standard RAG pipeline (ChromaDB)
├── settings_template.yaml   # GraphRAG config template
│
├── source/                  # Backend modules
│   ├── config.py            # Settings (.env)
│   ├── chat_engine.py       # Chat + SSE streaming
│   ├── query_engine.py      # GraphRAG queries
│   ├── indexer.py           # GraphRAG indexing
│   ├── preprocessor.py      # PDF/DOCX → TXT
│   ├── cache_manager.py     # Smart cache (memory + disk)
│   ├── embedding_batch.py   # Parallel embedding
│   ├── retry_handler.py     # Exponential backoff
│   ├── file_tracker.py      # File hash registry
│   ├── models.py            # Pydantic models
│   └── dashboard.py         # HTML dashboard
│
├── ui/                      # Streamlit pages
│   ├── api.py               # API helper
│   ├── sidebar.py           # Chat history sidebar
│   └── pages/               # Overview, Chat, Query, Graph, Compare
│
├── data/{domain}/           # Source documents
├── indexes/{domain}/        # GraphRAG output (Parquet, LanceDB)
├── db/standard_rag_chroma/  # ChromaDB storage
└── resources/uploads/       # Organized uploads (pdf/, docx/, txt/)
```

## API chính

```
GET    /health                              Health check
GET    /domains                             Liệt kê domains
POST   /domains/{domain}/upload             Upload files
POST   /chat                                Chat (SSE streaming)
POST   /query/local                         GraphRAG local search
POST   /query/global                        GraphRAG global search
POST   /api/compare-rag                     So sánh RAG vs GraphRAG
POST   /reindex/{domain}                    Re-index domain
GET    /graph/{domain}                      Knowledge Graph data
DELETE /domains/{domain}                    Xóa domain
```

Xem đầy đủ tại: http://localhost:8001/docs

## Xử lý sự cố

| Vấn đề | Giải pháp |
|---|---|
| Chat chậm/treo | OpenRouter free giới hạn ~5 req/min — đợi retry tự động hoặc nâng cấp key |
| GraphRAG indexing lỗi | Kiểm tra `LLM_API_KEY`, xóa `indexes/{domain}/` rồi reindex lại |
| Standard RAG không trả lời | Kiểm tra `db/standard_rag_chroma/{domain}/` có data, upload lại nếu cần |
| Out of memory | Giảm `num_threads` trong `settings_template.yaml`, giảm `batch_size` trong `standard_rag.py` |

---

<div align="center">

**Đồ án môn Phát triển Phần mềm Nâng cao**

Made with ❤️ using Python, FastAPI, GraphRAG & Streamlit

</div>
