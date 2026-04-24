# app/service/rebuild.py


from __future__ import annotations
from pathlib import Path
from tqdm import tqdm

from rag.document_loader import load_pdf
from rag.chunk_builder import build_chunks
from rag.embedding import GeminiEmbedder
from rag.vector_store import VectorStore


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw" / "document" / "pdf"
DB_DIR = ROOT_DIR / "db" / "chroma_db"


def rebuild_rag_database() -> dict:
    vector_store = VectorStore(path=str(DB_DIR))
    embedder = GeminiEmbedder()

    vector_store.reset()

    pdf_files = list(DATA_DIR.glob("**/*.pdf"))
    if not pdf_files:
        return {
            "status": "success",
            "message": f"No PDF files found in {DATA_DIR}",
            "files": 0,
            "chunks": 0,
        }

    total_chunks_processed = 0

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        pages = load_pdf(str(pdf_path))
        if not pages:
            continue

        chunks = build_chunks(pages, pdf_path.name)
        if not chunks:
            continue

        embedded_chunks = embedder.embed_chunks(chunks)
        if not embedded_chunks:
            continue

        vector_store.add_embeddings(embedded_chunks)
        total_chunks_processed += len(embedded_chunks)

    return {
        "status": "success",
        "message": "Rebuild completed",
        "files": len(pdf_files),
        "chunks": total_chunks_processed,
    }