from __future__ import annotations

from typing import Any, Dict, Optional

import chromadb


class Retriever:
    def __init__(self, path: str = "db/chroma_db", collection_name: str = "rag_documents") -> None:
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_collection(name=collection_name)

    def retrieve(
        self,
        query_embedding,
        k: int = 8,
        where: Optional[Dict[str, Any]] = None,
    ):
        if not query_embedding:
            raise ValueError("query_embedding is empty")

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k,
        }

        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        return results