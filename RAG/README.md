# DocRAG

DocRAG là một hệ thống Retrieval-Augmented Generation (RAG) cho phép bạn hỏi đáp (Q&A) trên các tài liệu PDF của riêng mình. Dự án sử dụng sức mạnh của các mô hình Gemini từ Google để hiểu và tạo ra câu trả lời dựa trên nội dung tài liệu của bạn.

## Tính năng

-   **Xử lý tài liệu PDF**: Tự động đọc, phân đoạn và xử lý văn bản từ các file PDF.
-   **Tích hợp Gemini**: Sử dụng các mô-hình Gemini mới nhất để tạo embeddings và sinh câu trả lời chất lượng cao.
-   **Vector Store hiệu quả**: Lưu trữ và truy vấn embeddings văn bản bằng ChromaDB.
-   **Truy cập linh hoạt**: Tương tác với hệ thống qua hai cách:
    -   Giao diện dòng lệnh (CLI) trực quan.
    -   API endpoint mạnh mẽ được xây dựng bằng FastAPI.
-   **Hỗ trợ trích dẫn (Citation)**: Tự động liên kết câu trả lời với nguồn tài liệu gốc.
-   **Theo dõi chi phí**: Theo dõi và thống kê lượng token sử dụng và chi phí cho các cuộc gọi đến Gemini API.

## Cách hoạt động

Kiến trúc của DocRAG tuân theo một luồng RAG cổ điển:

1.  **Load**: Các tài liệu PDF từ thư mục `data/raw/document/pdf` được tải vào hệ thống.
2.  **Chunk**: Nội dung văn bản của mỗi tài liệu được chia thành các đoạn (chunk) nhỏ hơn để dễ dàng xử lý.
3.  **Embed**: Mỗi chunk được chuyển thành một vector embedding bằng mô hình `text-embedding-004` của Gemini.
4.  **Store**: Các chunks và vector embeddings tương ứng được lưu trữ trong cơ sở dữ liệu vector ChromaDB.
5.  **Retrieve**: Khi người dùng đặt câu hỏi, hệ thống sẽ tạo embedding cho câu hỏi và sử dụng nó để tìm kiếm các chunk văn bản có liên quan nhất trong ChromaDB.
6.  **Generate**: Các chunk văn bản được truy xuất sẽ được dùng làm ngữ cảnh (context) và đưa vào một mô hình sinh văn bản của Gemini (ví dụ: `gemini-1.5-flash`) cùng với câu hỏi của người dùng. Mô hình sẽ tạo ra câu trả lời dựa trên ngữ cảnh này.

## Cấu trúc dự án

```
/
├─── data/raw/document/pdf/   # Nơi để các file PDF của bạn
├─── db/chroma_db/            # Thư mục lưu trữ của ChromaDB
├─── app/                     # Mã nguồn cho FastAPI
│   └─── api.py               # Định nghĩa các API endpoints
├─── rag/                     # Logic lõi của RAG pipeline
│   ├─── embedding.py         # Tạo embeddings
│   ├─── retriever.py         # Truy xuất văn bản
│   ├─── generator.py         # Sinh câu trả lời
│   └─── ...
├─── main.py                  # Chạy ứng dụng dưới dạng CLI
├─── populate_db.py           # Script để xử lý tài liệu và nạp vào DB
├─── requirements.txt         # Các thư viện Python cần thiết
└─── README.md
```

## Cài đặt và Thiết lập

**Yêu cầu**: Python 3.8+

**1. Clone repository**

```bash
git clone <URL_CUA_REPOSITORY>
cd DocRAG
```

**2. Tạo môi trường ảo và cài đặt dependencies**

```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Thiết lập API Key**

-   Tạo một file tên là `.env` ở thư mục gốc của dự án.
-   Lấy API key của bạn từ [Google AI Studio](https://aistudio.google.com/app/apikey).
-   Thêm key vào file `.env`:

```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

## Hướng dẫn sử dụng

**Bước 1: Thêm tài liệu**

-   Copy tất cả các file PDF bạn muốn hệ thống học vào thư mục `data/raw/document/pdf/`.

**Bước 2: Xử lý tài liệu và nạp vào cơ sở dữ liệu**

-   Chạy script `populate_db.py` để bắt đầu quá trình xử lý, tạo embedding và lưu trữ vào ChromaDB.

```bash
python populate_db.py
```

Quá trình này có thể mất một lúc tùy thuộc vào số lượng và kích thước của tài liệu. Bạn chỉ cần chạy lại script này khi có thêm tài liệu mới hoặc muốn cập nhật lại toàn bộ cơ sở dữ liệu.

**Bước 3: Chạy ứng dụng**

Bạn có thể tương tác với DocRAG qua CLI hoặc qua API.

### **Option A: Sử dụng giao diện dòng lệnh (CLI)**

Chạy `main.py` để bắt đầu một phiên chat trong terminal:

```bash
python main.py
```

Sau đó, bạn có thể bắt đầu đặt câu hỏi. Gõ `quit` hoặc `exit` để thoát.

### **Option B: Sử dụng API Server**

Khởi động server FastAPI bằng `uvicorn`:

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Server sẽ chạy tại `http://0.0.0.0:8000`.

**Endpoints:**

-   `GET /health`: Kiểm tra trạng thái của server.
-   `POST /ask`: Gửi câu hỏi và nhận câu trả lời.
-   `POST /rebuild-rag`: Chạy lại quá trình `populate_db` từ xa.

**Ví dụ sử dụng `curl`:**

```bash
# Hỏi một câu hỏi
curl -X POST http://0.0.0.0:8000/ask \
-H "Content-Type: application/json" \
-d '{"question": "Câu hỏi của bạn là gì?"}'

# Chạy lại quá trình xử lý tài liệu
curl -X POST http://0.0.0.0:8000/rebuild-rag
```

## Cấu hình

Bạn có thể thay đổi mô hình Gemini được sử dụng để sinh câu trả lời bằng cách thêm biến môi trường `GEMINI_GENERATE_MODEL` vào file `.env`.

**Ví dụ:**

```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
GEMINI_GENERATE_MODEL="gemini-1.5-pro-latest"
```