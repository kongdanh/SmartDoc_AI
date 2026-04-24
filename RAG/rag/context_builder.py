from __future__ import annotations


class ContextBuilder:
    def __init__(self, compact_mode: bool = True):
        """
        Initialize ContextBuilder.
        
        Args:
            compact_mode: If True, use minimal headers (faster, fewer tokens).
                         If False, use verbose headers with all metadata.
        """
        self.compact_mode = compact_mode

    def build(self, retrieved_docs) -> str:
        documents = retrieved_docs.get("documents", [[]])[0]
        metadatas = retrieved_docs.get("metadatas", [[]])[0]

        context_parts = []

        for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            source = meta.get("source", "unknown")
            page = meta.get("page", "N/A")

            if self.compact_mode:
                # Compact header: minimal metadata, fewer tokens
                header = f"[{i}] {source}:{page}"
            else:
                # Verbose header: all metadata (original behavior)
                section = meta.get("section_heading", "") or meta.get("raw_section_heading", "")
                raw_title = meta.get("raw_title", "") or meta.get("page_title", "")
                chunk_type = meta.get("chunk_type", "")
                block_types = ", ".join(meta.get("block_types", [])) if meta.get("block_types") else ""

                header = f"[DOC {i}] source={source} | page={page}"
                if raw_title:
                    header += f" | title={raw_title}"
                if section:
                    header += f" | section={section}"
                if chunk_type:
                    header += f" | type={chunk_type}"
                if block_types:
                    header += f" | blocks={block_types}"

            context_parts.append(f"{header}\ncontent:\n{doc}")

        return "\n\n".join(context_parts)