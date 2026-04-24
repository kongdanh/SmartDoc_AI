from __future__ import annotations

from rag.chunking import Chunker
from rag.metadata_builder import MetadataBuilder


def build_chunks(documents, file_name, chunking_strategy="auto"):
    chunker = Chunker()
    metadata_builder = MetadataBuilder()

    base_metadata = metadata_builder.build_metadata(
        filename=file_name,
        page_number=0,
        text=file_name,
        extra={
            "chunking_strategy": "production_pdf_pipeline",
            "dedup_enabled": True,
            "semantic_refine": True,
            "table_aware": True,
            "layout_aware": True,
            "structure_aware": True,
        },
    )

    base_metadata.pop("page", None)
    base_metadata.pop("source_label", None)

    chunks = chunker.chunk(
        content=documents,
        source_name=file_name,
        base_metadata=base_metadata,
    )

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["content"]
        page = chunk["metadata"].get("page", 0)

        merged_meta = metadata_builder.build_metadata(
            filename=file_name,
            page_number=page,
            chunk_index=i,
            text=text,
            extra=chunk["metadata"],
        )

        enriched.append(
            {
                "id": chunk["id"],
                "content": text,
                "metadata": merged_meta,
            }
        )

    return enriched