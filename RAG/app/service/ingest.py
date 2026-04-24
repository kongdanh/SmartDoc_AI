# app/service/ingest.py

from __future__ import annotations

from pathlib import Path
from typing import Dict
from rag.document_loader import load_pdf
from rag.chunk_builder import build_chunks
from rag.embedding import GeminiEmbedder
from rag.vector_store import VectorStore


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw" / "document" / "pdf"
DB_DIR = ROOT_DIR / "db" / "chroma_db"


def ingest_pdf_file(file_path: str) -> Dict:
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    embedder = GeminiEmbedder()
    vector_store = VectorStore(path=str(DB_DIR))

    pages = load_pdf(str(pdf_path))
    if not pages:
        return {
            "status": "failed",
            "message": "Không extract được nội dung từ PDF.",
            "file": pdf_path.name,
            "chunks": 0,
        }

    chunks = build_chunks(pages, pdf_path.name)
    if not chunks:
        return {
            "status": "failed",
            "message": "Không tạo được chunk từ tài liệu.",
            "file": pdf_path.name,
            "chunks": 0,
        }

    embedded_chunks = embedder.embed_chunks(chunks)
    if not embedded_chunks:
        return {
            "status": "failed",
            "message": "Không tạo được embedding.",
            "file": pdf_path.name,
            "chunks": 0,
        }

    vector_store.add_embeddings(embedded_chunks)

    return {
        "status": "success",
        "message": "Upload và ingest thành công.",
        "file": pdf_path.name,
        "chunks": len(embedded_chunks),
    }


def save_upload_file(upload_file, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / upload_file.filename

    with destination_path.open("wb") as f:
        f.write(upload_file.file.read())

    return destination_path


def save_and_ingest_upload(upload_file) -> Dict:
    saved_path = save_upload_file(upload_file, DATA_DIR)
    return ingest_pdf_file(str(saved_path))