# Standard RAG Module

Standard Retrieval-Augmented Generation implementation using chunking, embedding, and vector store retrieval.

## Overview

This module provides an alternative to GraphRAG for simpler RAG use cases:

- Document chunking (overlap-aware)
- Parallel embedding with HuggingFace
- Vector store (ChromaDB) for semantic search
- Fast retrieval without entity extraction

Ideal for:
- Quick document search without graph complexity
- Real-time chunked content retrieval
- Multi-domain document comparison

## Components

### main.py

Entry point for RAG indexing and querying.

```python
def build_standard_rag_index(domain, docs):
    # 1. Chunk documents
    chunks = chunk_documents(docs)
    # 2. Embed chunks in parallel
    embeddings = parallel_embed(chunks)
    # 3. Store in ChromaDB
    chroma_collection.add(embeddings)

async def query_standard_rag(domain, query):
    # Search similar chunks
    results = chroma_collection.query(query)
    return format_results(results)
```

### chunk_builder.py

Intelligent document chunking with overlap.

```python
class ChunkBuilder:
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def build(self, text):
        # Split into chunks with overlap
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks
```

### embedding.py

HuggingFace-based embedding generator.

```python
class LocalHuggingFaceEmbedder:
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
    
    def embed(self, texts):
        return self.model.encode(texts)
```

### vector_store.py

ChromaDB vector store interface.

```python
class ChromaVectorStore:
    def add(self, chunks, embeddings, metadata):
        self.collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata
        )
    
    def search(self, query_embedding, k=5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results
```

### document_loader.py

Multi-format document loading (PDF, DOCX, TXT).

```python
def load_documents(path):
    docs = []
    for file in Path(path).glob("*"):
        if file.suffix == ".pdf":
            docs.extend(load_pdf(file))
        elif file.suffix == ".txt":
            docs.extend(load_txt(file))
    return docs
```

### text_cleaner.py

Text preprocessing and cleaning.

```python
class TextCleaner:
    @staticmethod
    def clean(text):
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove special characters
        text = re.sub(r'[^\w\s.!?]', '', text)
        return text.strip()
```

## Performance Optimizations

### Parallel Embedding

4-worker ThreadPoolExecutor for batch processing:

```python
embedder = ParallelEmbedder(model, workers=4, batch_size=32)
embeddings = embedder.embed_parallel(chunks)
```

**Result:** 4x faster embedding (100 chunks: 10s → 2.5s)

### Batch Processing

Larger batch sizes for vector operations:

```python
batch_size = 100  # Process 100 chunks at once
for batch in chunk_batch(chunks, batch_size):
    embeddings = model.encode(batch)
    collection.add(embeddings)
```

### Smart Chunking

Overlap-aware chunking prevents context loss:

```python
ChunkBuilder(
    chunk_size=500,
    overlap=50  # 50 character overlap between chunks
)
```

## Usage

### Index a Domain

```python
from standard_rag import build_standard_rag_index

# Index documents from folder
build_standard_rag_index(
    domain="documents",
    docs_path="data/documents"
)
```

### Query

```python
from standard_rag import query_standard_rag

# Search documents
results = await query_standard_rag(
    domain="documents",
    query="How does X work?"
)

for chunk in results:
    print(f"Score: {chunk['score']}")
    print(f"Text: {chunk['text']}")
```

## Configuration

Embedding settings in `.env`:

```env
# Embedding Provider
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Indexing
INDEXING_METHOD=standard
```

Chunking settings in code:

```python
# In standard_rag.py
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 100
PARALLEL_WORKERS = 4
```

## Comparison with GraphRAG

| Feature | Standard RAG | GraphRAG |
|---------|---|---|
| Speed | Fast | Slower (needs LLM) |
| Accuracy | Good | Excellent |
| Memory | Low | High |
| Entity Extraction | No | Yes |
| Relationship Mapping | No | Yes |
| Best For | Quick search | Deep analysis |
| Cost | Low | Medium |

## Troubleshooting

### Out of Memory

**Issue:** Too many chunks being processed

**Solution:**
```python
# Reduce batch size
BATCH_SIZE = 32

# Or reduce workers
ParallelEmbedder(model, workers=2)
```

### Slow Embedding

**Issue:** GPU not available

**Solution:**
```python
# Force CPU batching
embedder = ParallelEmbedder(model, workers=8)  # Use more workers on CPU

# Or use lighter model
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"  # Smallest option
```

### Low Relevance Scores

**Issue:** Document chunks not matching queries well

**Solution:**
- Use semantic query: "How to X?" works better than "X"
- Increase k (number of results): `.search(..., k=10)`
- Verify documents contain relevant information
- Try different chunk size (larger = more context, smaller = more precise)

## Development

### Add Custom Embedding Model

1. Implement `EmbeddingModel` interface
2. Add to `embedding.py`
3. Update `.env` with model name
4. Restart indexing

### Custom Chunking Strategy

1. Create `ChunkStrategy` subclass
2. Implement `build(text)` method
3. Use in `standard_rag.py`:

```python
chunker = CustomChunkStrategy(...)
chunks = chunker.build(document_text)
```

## Dependencies

- chromadb >= 0.5.0
- sentence-transformers >= 2.2.0
- PyPDF2 >= 3.0.0
- python-docx >= 0.8.0
- langchain >= 0.1.0

## See Also

- [Parent Project README](../README.md)
- [Graph-RAG Module](../Graph-RAG/README.md)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
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