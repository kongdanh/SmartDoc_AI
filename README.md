# 🧠 SmartDoc AI — Intelligent Document Analysis with GraphRAG

<div align="center">

![SmartDoc AI](https://img.shields.io/badge/SmartDoc_AI-v1.0-6c8aff?style=for-the-badge&logo=brain&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![GraphRAG](https://img.shields.io/badge/Microsoft-GraphRAG-4caf84?style=for-the-badge&logo=microsoft&logoColor=white)

**Hệ thống Quản lý và Khai thác Tri thức thông minh sử dụng GraphRAG**

*Đồ án môn Phát triển phần mềm nâng cao — Đại học Công nghệ Thông tin*

</div>

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Tính năng chính](#-tính-năng-chính)
- [So sánh RAG vs GraphRAG](#-so-sánh-rag-vs-graphrag)
- [Cài đặt](#-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [API Documentation](#-api-documentation)
- [Quy trình hoạt động](#-quy-trình-hoạt-động)

---

## 🎯 Giới thiệu

**SmartDoc AI** là hệ thống quản lý và khai thác tri thức thông minh, kết hợp hai công nghệ AI tiên tiến:

1. **Microsoft GraphRAG** — Xây dựng Knowledge Graph từ tài liệu, trích xuất entities & relationships, tạo community reports để trả lời câu hỏi phức tạp, tổng hợp.

2. **Standard RAG (FAISS)** — Sử dụng Vector Database truyền thống với FAISS indexing, phù hợp cho câu hỏi cụ thể, trực tiếp.

Hệ thống cho phép người dùng **upload tài liệu** (PDF, DOCX, TXT, code...), tự động xây dựng **Knowledge Graph**, và **so sánh trực tiếp** kết quả giữa hai phương pháp RAG.

### Bối cảnh & Mục tiêu

Trong bối cảnh bùng nổ dữ liệu phi cấu trúc, việc khai thác tri thức từ tài liệu đòi hỏi các phương pháp tiên tiến hơn tìm kiếm vector đơn thuần. GraphRAG giải quyết điểm yếu của RAG truyền thống bằng cách:

- **Hiểu ngữ cảnh rộng**: Nhờ Knowledge Graph, hệ thống hiểu mối quan hệ giữa các khái niệm
- **Trả lời câu hỏi tổng hợp**: Community reports cung cấp cái nhìn toàn cảnh
- **Truy vấn đa phương pháp**: Local, Global, DRIFT search phục vụ nhiều nhu cầu khác nhau

---

## 🏗 Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                        │
│  ┌─────────────────┐    ┌──────────────────────────┐     │
│  │  Streamlit UI   │    │   HTML Dashboard (D3.js) │     │
│  │  (app_ui.py)    │    │   (dashboard.html)       │     │
│  └────────┬────────┘    └────────────┬─────────────┘     │
│           │         REST API         │                   │
└───────────┼──────────────────────────┼───────────────────┘
            │                          │
┌───────────┼──────────────────────────┼───────────────────┐
│           ▼          BACKEND LAYER   ▼                   │
│  ┌────────────────────────────────────────────────┐      │
│  │             FastAPI Server (main.py)           │      │
│  │  ┌──────────┐ ┌──────────┐ ┌───────────────┐  │      │
│  │  │  Chat    │ │  Query   │ │  Compare RAG  │  │      │
│  │  │  Engine  │ │  Engine  │ │  Engine       │  │      │
│  │  └──────────┘ └──────────┘ └───────────────┘  │      │
│  └────────────────────────────────────────────────┘      │
│                                                          │
│  ┌──────────────────────┐  ┌──────────────────────────┐  │
│  │   GraphRAG Pipeline  │  │  Standard RAG Pipeline   │  │
│  │  ┌────────────────┐  │  │  ┌──────────────────┐    │  │
│  │  │  Text Chunking │  │  │  │  Text Splitting  │    │  │
│  │  │  Entity Extract│  │  │  │  FAISS Indexing   │    │  │
│  │  │  Graph Build   │  │  │  │  Vector Search    │    │  │
│  │  │  Community     │  │  │  │  LLM Generation   │    │  │
│  │  │  Reports       │  │  │  └──────────────────┘    │  │
│  │  └────────────────┘  │  └──────────────────────────┘  │
│  └──────────────────────┘                                │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │              DATA & INDEX LAYER                   │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │    │
│  │  │  Parquet │  │  LanceDB │  │  FAISS Index │   │    │
│  │  │  Files   │  │  Vectors │  │  (Vector DB) │   │    │
│  │  └──────────┘  └──────────┘  └──────────────┘   │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
            │                          │
┌───────────┼──────────────────────────┼───────────────────┐
│           ▼        LLM LAYER         ▼                   │
│  ┌────────────────┐       ┌──────────────────────┐       │
│  │  Groq Cloud    │       │  HuggingFace Local   │       │
│  │  (LLaMA 3.1)   │       │  (MiniLM Embeddings) │       │
│  └────────────────┘       └──────────────────────┘       │
└──────────────────────────────────────────────────────────┘
```

---

## 🛠 Công nghệ sử dụng

### Backend
| Công nghệ | Vai trò | Phiên bản |
|-----------|---------|-----------|
| **Python** | Ngôn ngữ chính | 3.10+ |
| **FastAPI** | REST API Framework | ≥0.115 |
| **Uvicorn** | ASGI Server | ≥0.30 |
| **Pydantic** | Data Validation | ≥2.0 |

### AI & Machine Learning
| Công nghệ | Vai trò | Phiên bản |
|-----------|---------|-----------|
| **Microsoft GraphRAG** | Knowledge Graph RAG | ≥3.0 |
| **LangChain** | Standard RAG Pipeline | ≥0.2 |
| **FAISS** | Vector Database | ≥1.7 |
| **Groq (LLaMA 3.1 70B)** | Large Language Model | API |
| **HuggingFace (MiniLM)** | Text Embeddings (Local) | sentence-transformers |
| **LiteLLM** | LLM Gateway | Built-in |

### Frontend
| Công nghệ | Vai trò | Phiên bản |
|-----------|---------|-----------|
| **Streamlit** | Interactive UI Dashboard | ≥1.30 |
| **Streamlit-AGraph** | Knowledge Graph Visualization | ≥0.0.45 |
| **D3.js** | Graph Visualization (HTML) | v7 |
| **Marked.js** | Markdown Rendering | v12 |

### Document Processing
| Công nghệ | Vai trò | Phiên bản |
|-----------|---------|-----------|
| **PyMuPDF (fitz)** | PDF Text Extraction | ≥1.24 |
| **python-docx** | DOCX Text Extraction | ≥1.0 |
| **LanceDB** | Vector Storage for GraphRAG | Built-in |
| **Pandas** | Data Processing | ≥2.0 |

---

## ✨ Tính năng chính

### 1. 📊 Overview Dashboard
- Hiển thị tổng quan hệ thống: số domains, trạng thái ready, số files đã index
- Quản lý domains: xem files, re-index, xóa domain
- Xóa file riêng lẻ trong mỗi domain

### 2. 💬 AI Chat (GraphRAG-Augmented)
- Chat streaming real-time với SSE (Server-Sent Events)
- Context retrieval từ Knowledge Graph (entities, relationships, sources)
- Hỗ trợ multi-turn conversation (nhiều lượt hội thoại)
- Lịch sử chat: xem, tiếp tục, xóa từng session hoặc xóa toàn bộ
- Markdown rendering trong câu trả lời

### 3. 🔍 Advanced Query
- **Local Search**: Tìm kiếm entity-focused, trả lời câu hỏi cụ thể
- **Global Search**: Phân tích dataset-wide, câu hỏi tổng hợp
- **DRIFT Search**: Hybrid (Global → Local drill-down)
- **Direct Search**: Tìm kiếm trực tiếp trong Parquet files, không cần LLM
- Hiển thị entities, relationships, sources riêng biệt

### 4. 🕸️ Knowledge Graph Visualization
- Trực quan hóa đồ thị tri thức với D3.js / Streamlit-AGraph
- Force-directed layout với drag & zoom
- Color coding theo degree (red: high, blue: medium, green: low)
- Tooltip hiển thị entity details
- Entity data table

### 5. 📤 File Upload & Management
- Upload đa file: PDF, DOCX, TXT, Markdown, CSV, JSON, Code files
- Tự động chuyển đổi PDF → TXT, DOCX → TXT
- Song song: nạp vào FAISS (Standard RAG) + GraphRAG indexing
- Quản lý files: xem, xóa file riêng lẻ, xóa toàn bộ domain
- Auto re-index sau upload

### 6. ⚖️ Compare RAG vs GraphRAG
- So sánh song song kết quả giữa Standard RAG và GraphRAG
- Chạy cả hai AI đồng thời (parallel) tiết kiệm 50% thời gian
- Hiển thị bảng so sánh 2 cột
- Phân tích: độ dài câu trả lời, thời gian phản hồi

---

## 📊 So sánh RAG vs GraphRAG

| Tiêu chí | Standard RAG (FAISS) | Microsoft GraphRAG |
|----------|---------------------|-------------------|
| **Phương pháp** | Vector similarity search | Knowledge Graph + Community Reports |
| **Embedding** | HuggingFace MiniLM (local) | HuggingFace MiniLM (local) |
| **Loại câu hỏi phù hợp** | Cụ thể, trực tiếp | Tổng hợp, phức tạp |
| **Tốc độ trả lời** | Nhanh (chỉ vector search + LLM) | Chậm hơn (graph traversal + LLM) |
| **Hiểu ngữ cảnh** | Cục bộ (chỉ chunks tương tự) | Toàn cục (relationships + communities) |
| **Yêu cầu indexing** | FAISS index (nhanh) | Full graph extraction (lâu hơn) |
| **Storage** | FAISS binary files | Parquet + LanceDB |

---

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.10+
- 8GB RAM (tối thiểu)
- GPU không bắt buộc (embedding chạy CPU)

### Bước 1: Clone Repository

```bash
git clone <repository-url>
cd code
```

### Bước 2: Tạo Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Bước 3: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình (.env)

Chỉnh sửa file `.env` theo nhu cầu:

```env
# ─── LLM (Groq Cloud - miễn phí, siêu nhanh) ───
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-70b-versatile
LLM_API_KEY=your_groq_api_key_here
LLM_RPM=10

# ─── Embedding (HuggingFace - chạy local, miễn phí) ───
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# ─── Server ───
DATA_DIR=./data
INDEX_DIR=./indexes
SERVER_PORT=8001
```

> **Lưu ý**: Đăng ký API key Groq miễn phí tại [console.groq.com](https://console.groq.com)

### Bước 5: Khởi chạy

**Terminal 1 — Backend (FastAPI):**
```bash
python main.py
```

**Terminal 2 — Frontend (Streamlit):**
```bash
streamlit run app_ui.py
```

Truy cập:
- **Streamlit UI**: http://localhost:8501
- **FastAPI Dashboard**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

---

## 📖 Hướng dẫn sử dụng

### 1. Upload tài liệu

1. Mở tab **📤 Upload Files**
2. Nhập tên domain (VD: `tai_lieu_cntt`)
3. Kéo thả hoặc chọn files (PDF, DOCX, TXT...)
4. Nhấn **Upload & Process**
5. Hệ thống tự động:
   - Chuyển đổi PDF/DOCX → TXT
   - Nạp vào FAISS (Standard RAG)
   - Chạy GraphRAG indexing (tạo Knowledge Graph)

### 2. Chat với AI

1. Mở tab **💬 Chat**
2. Chọn domain đã index
3. Đặt câu hỏi — AI sẽ trả lời dựa trên knowledge base
4. Lịch sử chat hiển thị trong sidebar

### 3. So sánh RAG

1. Mở tab **⚖️ Compare RAG**
2. Chọn domain, nhập câu hỏi
3. Nhấn **Run Comparison**
4. Xem kết quả song song giữa Standard RAG và GraphRAG

### 4. Trực quan Knowledge Graph

1. Mở tab **🕸️ Knowledge Graph**
2. Chọn domain → **Load Graph**
3. Kéo thả, zoom để khám phá mạng lưới tri thức

---

## 📁 Cấu trúc thư mục

```
SmartDoc-AI/
├── main.py                  # FastAPI backend server
├── app_ui.py                # Streamlit frontend UI
├── standard_rag.py          # Standard RAG pipeline (FAISS + LangChain)
├── requirements.txt         # Python dependencies
├── .env                     # Environment configuration
├── settings_template.yaml   # GraphRAG settings template
│
├── source/                  # Core modules
│   ├── __init__.py
│   ├── config.py            # Settings management (Pydantic)
│   ├── chat_engine.py       # Chat with RAG augmentation + streaming
│   ├── query_engine.py      # GraphRAG query methods (local/global/drift)
│   ├── indexer.py           # Auto-indexing engine for GraphRAG
│   ├── preprocessor.py      # File conversion (PDF, DOCX, TXT...)
│   ├── file_tracker.py      # Hash-based file change detection
│   ├── models.py            # Pydantic request/response models
│   ├── dashboard.py         # HTML dashboard router
│   └── dashboard.html       # Web-based admin dashboard (D3.js)
│
├── data/                    # Source documents (organized by domain)
│   ├── test1/               # Example domain
│   └── .../
│
├── indexes/                 # GraphRAG indexed data
│   ├── test1/
│   │   ├── input/           # Preprocessed text files
│   │   ├── output/          # Parquet files (entities, relationships...)
│   │   ├── cache/           # GraphRAG cache
│   │   └── settings.yaml    # Domain-specific GraphRAG config
│   └── file_registry.json   # File hash registry
│
├── faiss_index/             # FAISS vector database
│   ├── index.faiss
│   └── index.pkl
│
└── pdf_uploads/             # Temporary PDF upload storage
```

---

## 🔌 API Documentation

### Domain Management

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| `GET` | `/domains` | Liệt kê tất cả domains |
| `GET` | `/domains/{domain}/files` | Liệt kê files trong domain |
| `POST` | `/domains/{domain}/upload` | Upload files vào domain |
| `DELETE` | `/domains/{domain}` | Xóa toàn bộ domain |
| `DELETE` | `/domains/{domain}/files/{filename}` | Xóa file riêng lẻ |

### Query

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| `POST` | `/query/local` | Local search (entity-focused) |
| `POST` | `/query/global` | Global search (dataset-wide) |
| `POST` | `/query/drift` | DRIFT search (hybrid) |
| `POST` | `/query/search` | Direct text search (no LLM) |

### Chat

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| `POST` | `/chat` | Gửi tin nhắn (SSE streaming) |
| `GET` | `/chat/sessions` | Liệt kê chat sessions |
| `GET` | `/chat/sessions/{id}` | Chi tiết session |
| `DELETE` | `/chat/sessions/{id}` | Xóa session |

### Compare & System

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| `POST` | `/api/compare-rag` | So sánh RAG vs GraphRAG |
| `GET` | `/graph/{domain}` | Knowledge Graph data |
| `GET` | `/status` | Server status |
| `POST` | `/reindex/{domain}` | Re-index domain |
| `POST` | `/reindex` | Re-index tất cả |

### Ví dụ gọi API

```python
import requests

# So sánh RAG vs GraphRAG
response = requests.post("http://localhost:8001/api/compare-rag", json={
    "query": "Tóm tắt nội dung chính của tài liệu",
    "domain": "test1"
})
result = response.json()
print("Standard RAG:", result["standard_rag_answer"])
print("GraphRAG:", result["graph_rag_answer"])
```

---

## ⚙️ Quy trình hoạt động

### 1. Luồng Upload & Indexing

```
User Upload File
       │
       ▼
  ┌─────────┐
  │ FastAPI  │──── Lưu file gốc vào data/{domain}/
  │ Server   │
  └────┬────┘
       │
       ├───── PDF?  ──── PyMuPDF ──── Trích xuất text ──── TXT
       ├───── DOCX? ──── python-docx ─ Trích xuất text ──── TXT
       └───── TXT?  ──── Đọc trực tiếp
       │
       ▼
  ┌──────────────┐          ┌──────────────────┐
  │ FAISS Index  │          │  GraphRAG Index   │
  │ (Standard)   │          │  (Knowledge Graph)│
  │              │          │                   │
  │ Text Split   │          │ Entity Extraction │
  │    1000 chars │          │ Relationship     │
  │ HuggingFace  │          │ Community Reports│
  │ Embedding    │          │ Parquet Storage  │
  │ FAISS Store  │          │ LanceDB Vectors  │
  └──────────────┘          └──────────────────┘
```

### 2. Luồng Query / Chat

```
User Question
       │
       ▼
  ┌───────────────────────────────────────┐
  │          Context Retrieval            │
  │                                       │
  │  1. Search entities (text matching)   │
  │  2. Search relationships              │
  │  3. Search community reports          │
  │  4. Fallback: raw text_units          │
  └───────────────┬───────────────────────┘
                  │
                  ▼
  ┌───────────────────────────────────────┐
  │    LLM (Groq - LLaMA 3.1 70B)       │
  │                                       │
  │  System Prompt + Context + History    │
  │         → Streaming Response          │
  └───────────────────────────────────────┘
```

### 3. Luồng Compare RAG

```
User Question
       │
       ▼
  ┌──────────────────────────┐
  │   asyncio.gather()       │   ← Chạy song song
  │                          │
  │  ┌────────────────────┐  │
  │  │  Standard RAG      │  │
  │  │  FAISS Search      │  │
  │  │  + LLM Generate    │  │
  │  └────────────────────┘  │
  │                          │
  │  ┌────────────────────┐  │
  │  │  GraphRAG          │  │
  │  │  Local Search CLI  │  │
  │  │  + LLM Generate    │  │
  │  └────────────────────┘  │
  └──────────────┬───────────┘
                 │
                 ▼
  ┌──────────────────────────┐
  │   JSON Response          │
  │   {                      │
  │     question,            │
  │     standard_rag_answer, │
  │     graph_rag_answer     │
  │   }                      │
  └──────────────────────────┘
```

---

## 👨‍💻 Tác giả

**Đồ án môn Phát triển Phần mềm Nâng cao**

---

## 📄 License

This project is for educational purposes — University Capstone Project.

---

<div align="center">

**🧠 SmartDoc AI** — *Transforming Documents into Knowledge*

Made with ❤️ using Python, FastAPI, GraphRAG & Streamlit

</div>
