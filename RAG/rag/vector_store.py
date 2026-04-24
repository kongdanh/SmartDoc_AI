from __future__ import annotations

from typing import Any, Dict, List

import chromadb


def _sanitize_metadata_value(value: Any):
    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple, set)):
        return " | ".join(str(v) for v in value)

    if isinstance(value, dict):
        return str(value)

    return str(value)


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for key, value in metadata.items():
        sanitized[str(key)] = _sanitize_metadata_value(value)
    return sanitized


class VectorStore:
    def __init__(self, path: str = "db/chroma_db", collection_name: str = "rag_documents") -> None:
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def reset(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print("ChromaDB collection reset.")

    def add_embeddings(self, embedded_chunks: List[Dict[str, Any]]) -> None:
        if not embedded_chunks:
            print("No embeddings to add.")
            return

        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for chunk in embedded_chunks:
            documents.append(chunk["content"])
            embeddings.append(chunk["embedding"])
            metadatas.append(_sanitize_metadata(chunk["metadata"]))
            ids.append(chunk["id"])

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        print(f"Stored {len(documents)} chunks in ChromaDB")