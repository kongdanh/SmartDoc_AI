from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from rag.document_loader import load_pdf
from rag.chunk_builder import build_chunks
from rag.embedding import GeminiEmbedder
from rag.vector_store import VectorStore


ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "raw" / "document" / "pdf"
DB_DIR = ROOT_DIR / "db" / "chroma_db"


def main() -> None:
    print("Initializing components...")
    vector_store = VectorStore(path=str(DB_DIR))
    embedder = GeminiEmbedder()

    print("Resetting ChromaDB collection...")
    vector_store.reset()

    pdf_files = list(DATA_DIR.glob("**/*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}. Exiting.")
        return

    print(f"Found {len(pdf_files)} PDF document(s) to process.")

    total_chunks_processed = 0

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            print(f"\nProcessing file: {pdf_path.name}")

            pages = load_pdf(str(pdf_path))
            if not pages:
                print(f"Could not extract text from {pdf_path.name}. Skipping.")
                continue

            chunks = build_chunks(pages, pdf_path.name)
            if not chunks:
                print(f"No chunks created for {pdf_path.name}. Skipping.")
                continue

            print(f"Created {len(chunks)} chunks.")

            print("Generating embeddings...")
            embedded_chunks = embedder.embed_chunks(chunks)

            print("Adding chunks to ChromaDB...")
            vector_store.add_embeddings(embedded_chunks)

            total_chunks_processed += len(embedded_chunks)

        except Exception as e:
            print(f"An error occurred while processing {pdf_path.name}: {e}")

    print("\n--- RAG Database Population Complete ---")
    print(f"Total chunks processed and stored: {total_chunks_processed}")


if __name__ == "__main__":
    main()