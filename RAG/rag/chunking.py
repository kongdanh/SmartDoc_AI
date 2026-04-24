from __future__ import annotations

import ast
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rag.embedding import GeminiEmbedder
from rag.tokenizer import Tokenizer


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.chunk_id,
            "content": self.text,
            "metadata": self.metadata,
        }


class Chunker:
    def __init__(
        self,
        chunk_size: int = 450,
        overlap: int = 60,
        semantic_max_tokens: int = 520,
        semantic_min_tokens: int = 120,
        dedup_similarity: float = 0.96,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.semantic_max_tokens = semantic_max_tokens
        self.semantic_min_tokens = semantic_min_tokens
        self.dedup_similarity = dedup_similarity

        self.tokenizer = Tokenizer()
        self.embedder = GeminiEmbedder()

    def chunk(
        self,
        content: Any,
        source_name: str = "",
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        base_metadata = dict(base_metadata or {})
        if source_name:
            base_metadata.setdefault("source_name", source_name)

        if isinstance(content, str):
            strategy = self.auto_detect_strategy(content)
            if strategy == "function":
                chunks = self._chunk_code(content, base_metadata)
            elif strategy == "structure":
                chunks = self._chunk_text_structure_first(content, base_metadata)
            elif strategy == "semantic":
                chunks = self._chunk_text_semantic_first(content, base_metadata)
            else:
                chunks = self._chunk_plain_text(content, base_metadata)

            return [c.to_dict() for c in self._deduplicate(chunks)]

        if isinstance(content, list):
            chunks = self.chunk_pdf_document(content, base_metadata=base_metadata)
            return [c.to_dict() for c in chunks]

        raise TypeError("content must be either str or list[page_dict]")

    def chunk_pdf_document(
        self,
        pages: Sequence[Dict[str, Any]],
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        base_metadata = dict(base_metadata or {})
        all_chunks: List[Chunk] = []

        for page in pages:
            all_chunks.extend(self._chunk_single_page(page, base_metadata))

        all_chunks = self._semantic_refine_large_chunks(all_chunks)
        all_chunks = self._deduplicate(all_chunks)
        return all_chunks

    def auto_detect_strategy(self, text: str) -> str:
        try:
            tree = ast.parse(text)
            has_blocks = any(
                isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                for node in ast.walk(tree)
            )
            if has_blocks:
                return "function"
        except SyntaxError:
            pass

        heading_count = len(re.findall(r"(?m)^#{1,6}\s+.+$", text))
        paragraph_count = len(re.findall(r"\n\s*\n", text))
        line_count = len(text.splitlines())

        if heading_count >= 2 or paragraph_count >= 3:
            return "structure"

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentences) >= 10 and len(text) >= 1000:
            return "semantic"

        if line_count > 20 and any(len(line.strip()) < 80 for line in text.splitlines()):
            return "structure"

        return "text"

    def _chunk_single_page(
        self,
        page: Dict[str, Any],
        base_metadata: Dict[str, Any],
    ) -> List[Chunk]:
        page_no = page.get("page")
        page_meta = dict(base_metadata)
        page_meta.update(page.get("metadata", {}))
        page_meta["page"] = page_no

        if page.get("title"):
            page_meta["page_title"] = page["title"]
            page_meta["raw_title"] = page["title"]

        blocks = list(page.get("blocks", []))
        if not blocks:
            text = self._clean_text(page.get("text", ""))
            if not text:
                return []
            return self._make_chunks_from_text(
                text=text,
                metadata={**page_meta, "chunk_type": "page"},
                parent_id=f"page:{page_no}",
            )

        ordered_blocks = self._sort_blocks_layout(blocks)

        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []

        for block in ordered_blocks:
            block_type = block.get("type", "text")

            if block_type == "table":
                if current:
                    groups.append(current)
                    current = []
                groups.append([block])
                continue

            if block_type == "heading" and current:
                groups.append(current)
                current = [block]
                continue

            if current and self._should_start_new_group(current[-1], block):
                groups.append(current)
                current = [block]
                continue

            current.append(block)

        if current:
            groups.append(current)

        chunks: List[Chunk] = []
        for index, group in enumerate(groups):
            group_type = self._infer_group_type(group)
            group_text = self._render_group_text(group)
            if not group_text:
                continue

            group_meta = dict(page_meta)
            group_meta.update(self._group_metadata(group))
            group_meta["group_index"] = index
            group_meta["chunk_type"] = group_type

            chunks.extend(
                self._make_chunks_from_text(
                    text=group_text,
                    metadata=group_meta,
                    parent_id=f"page:{page_no}:group:{index}",
                    prefer_single=(group_type == "table"),
                )
            )

        return chunks

    def _sort_blocks_layout(self, blocks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def sort_key(block: Dict[str, Any]) -> Tuple[float, int, float]:
            bbox = block.get("bbox") or [0, 0, 0, 0]
            x0, y0, _, _ = bbox
            col = block.get("column", 0)
            return (round(y0, 1), col, round(x0, 1))

        return sorted(blocks, key=sort_key)

    def _should_start_new_group(self, prev_block: Dict[str, Any], next_block: Dict[str, Any]) -> bool:
        prev_col = prev_block.get("column")
        next_col = next_block.get("column")
        prev_type = prev_block.get("type", "text")
        next_type = next_block.get("type", "text")

        if prev_type == "image" and next_type in {"text", "caption", "list"}:
            return False

        if prev_col is not None and next_col is not None and prev_col != next_col:
            return True

        prev_bbox = prev_block.get("bbox")
        next_bbox = next_block.get("bbox")
        if prev_bbox and next_bbox:
            _, _, _, prev_y1 = prev_bbox
            _, next_y0, _, _ = next_bbox
            if (next_y0 - prev_y1) > 80:
                return True

        return False

    def _infer_group_type(self, group: Sequence[Dict[str, Any]]) -> str:
        types = [b.get("type", "text") for b in group]
        if len(types) == 1 and types[0] == "table":
            return "table"
        if "heading" in types:
            return "section"
        if "image" in types and any(t in {"text", "caption", "list"} for t in types):
            return "visual_text"
        if "list" in types:
            return "list_block"
        return "text"

    def _group_metadata(self, group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        block_types = [b.get("type", "text") for b in group]
        cols = [b.get("column") for b in group if b.get("column") is not None]

        meta: Dict[str, Any] = {
            "block_types": sorted(set(block_types)),
        }

        if cols:
            meta["columns"] = sorted(set(cols))

        heading = next(
            (self._clean_text(b.get("text", "")) for b in group if b.get("type") == "heading"),
            "",
        )
        if heading:
            meta["section_heading"] = heading
            meta["raw_section_heading"] = heading

        return meta

    def _render_group_text(self, group: Sequence[Dict[str, Any]]) -> str:
        parts: List[str] = []

        for block in group:
            block_type = block.get("type", "text")
            text = self._clean_text(block.get("text", ""))

            if not text and block_type != "image":
                continue

            if block_type == "heading":
                parts.append(f"[HEADING] {text}")
            elif block_type == "table":
                parts.append(f"[TABLE]\n{text}")
            elif block_type == "caption":
                parts.append(f"[CAPTION] {text}")
            elif block_type == "image":
                alt_text = self._clean_text(block.get("alt_text", ""))
                if alt_text:
                    parts.append(f"[IMAGE] {alt_text}")
            elif block_type == "list":
                parts.append(f"[LIST] {text}")
            else:
                parts.append(text)

        return "\n".join(parts).strip()

    def _make_chunks_from_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        parent_id: str,
        prefer_single: bool = False,
    ) -> List[Chunk]:
        text = self._clean_text(text)
        if not text:
            return []

        tokens = self.tokenizer.encode(text)

        if prefer_single and len(tokens) <= self.semantic_max_tokens * 2:
            return [self._build_chunk(text, metadata, parent_id, 0)]

        if len(tokens) <= self.chunk_size:
            return [self._build_chunk(text, metadata, parent_id, 0)]

        pieces = self._split_text_semantically(
            text=text,
            max_tokens=self.chunk_size,
            min_tokens=min(self.semantic_min_tokens, self.chunk_size // 2),
            overlap=self.overlap,
        )
        return [self._build_chunk(piece, metadata, parent_id, i) for i, piece in enumerate(pieces)]

    def _chunk_plain_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        return self._make_chunks_from_text(
            text=text,
            metadata={**metadata, "chunk_type": "plain_text"},
            parent_id="plain_text",
        )

    def _chunk_text_structure_first(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        sections = re.split(r"(?m)(?=^#{1,6}\s+.+$)", text)
        if len(sections) <= 1:
            sections = re.split(r"\n\s*\n+", text)

        chunks: List[Chunk] = []
        for i, section in enumerate(filter(None, map(str.strip, sections))):
            chunks.extend(
                self._make_chunks_from_text(
                    text=section,
                    metadata={**metadata, "chunk_type": "structured_text"},
                    parent_id=f"structured:{i}",
                )
            )
        return chunks

    def _chunk_text_semantic_first(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        pieces = self._split_text_semantically(
            text=text,
            max_tokens=self.chunk_size,
            min_tokens=min(self.semantic_min_tokens, self.chunk_size // 2),
            overlap=self.overlap,
        )
        return [
            self._build_chunk(piece, {**metadata, "chunk_type": "semantic_text"}, "semantic", i)
            for i, piece in enumerate(pieces)
        ]

    def _chunk_code(self, code: str, metadata: Dict[str, Any]) -> List[Chunk]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._chunk_plain_text(code, metadata)

        lines = code.splitlines()
        chunks: List[Chunk] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                continue

            start = max(node.lineno - 1, 0)
            end = max(getattr(node, "end_lineno", node.lineno) - 1, start)
            block = "\n".join(lines[start : end + 1]).strip()
            if not block:
                continue

            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append(
                self._build_chunk(
                    block,
                    {
                        **metadata,
                        "chunk_type": "code",
                        "code_kind": kind,
                        "symbol": getattr(node, "name", ""),
                        "start_line": start + 1,
                        "end_line": end + 1,
                    },
                    parent_id="code",
                    index=len(chunks),
                )
            )

        if not chunks:
            return self._chunk_plain_text(code, metadata)
        return chunks

    def _semantic_refine_large_chunks(self, chunks: Sequence[Chunk]) -> List[Chunk]:
        refined: List[Chunk] = []
        for chunk in chunks:
            token_count = len(self.tokenizer.encode(chunk.text))
            if token_count <= self.semantic_max_tokens:
                refined.append(chunk)
                continue

            pieces = self._split_text_semantically(
                text=chunk.text,
                max_tokens=self.semantic_max_tokens,
                min_tokens=self.semantic_min_tokens,
                overlap=self.overlap,
            )

            if len(pieces) <= 1:
                refined.append(chunk)
                continue

            for i, piece in enumerate(pieces):
                meta = dict(chunk.metadata)
                meta["refined_from"] = chunk.chunk_id
                meta["refine_index"] = i
                refined.append(self._build_chunk(piece, meta, chunk.chunk_id, i))

        return refined

    def _split_text_semantically(
        self,
        text: str,
        max_tokens: int,
        min_tokens: int,
        overlap: int,
    ) -> List[str]:
        units = self._sentence_units(text)
        if len(units) <= 1:
            return self._sliding_token_split(text, max_tokens=max_tokens, overlap=overlap)

        embeddings = [self.embedder.embed_text(unit) for unit in units]
        similarities = []
        for i in range(len(embeddings) - 1):
            similarities.append(self._cosine_similarity(embeddings[i], embeddings[i + 1]))

        threshold = self._dynamic_similarity_threshold(similarities)

        chunks: List[str] = []
        current_units: List[str] = []
        current_tokens = 0
        i = 0

        while i < len(units):
            unit = units[i]
            unit_tokens = len(self.tokenizer.encode(unit))

            if current_units and current_tokens + unit_tokens > max_tokens:
                chunks.append(" ".join(current_units).strip())
                current_units, current_tokens = self._semantic_overlap_tail(current_units, overlap)
                continue

            current_units.append(unit)
            current_tokens += unit_tokens

            next_is_break = False
            if i < len(similarities):
                low_similarity = similarities[i] < threshold
                large_enough = current_tokens >= min_tokens
                near_budget = current_tokens >= int(max_tokens * 0.8)
                next_is_break = low_similarity and (large_enough or near_budget)

            if next_is_break:
                chunks.append(" ".join(current_units).strip())
                current_units, current_tokens = self._semantic_overlap_tail(current_units, overlap)

            i += 1

        if current_units:
            chunks.append(" ".join(current_units).strip())

        chunks = [c for c in chunks if c.strip()]
        if not chunks:
            return self._sliding_token_split(text, max_tokens=max_tokens, overlap=overlap)
        return chunks

    def _sentence_units(self, text: str) -> List[str]:
        text = self._clean_text(text)
        if not text:
            return []

        parts = re.split(r"(?<=[.!?;:])\s+|\n+", text)
        units = [p.strip() for p in parts if p and p.strip()]
        return units if units else [text]

    def _semantic_overlap_tail(self, units: Sequence[str], overlap: int):
        if overlap <= 0 or not units:
            return [], 0

        tail: List[str] = []
        count = 0
        for unit in reversed(units):
            unit_tokens = len(self.tokenizer.encode(unit))
            if count + unit_tokens > overlap and tail:
                break
            tail.insert(0, unit)
            count += unit_tokens

        return tail, count

    def _sliding_token_split(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks: List[str] = []
        start = 0
        step = max(1, max_tokens - max(overlap, 0))

        while start < len(tokens):
            end = start + max_tokens
            piece = self.tokenizer.decode(tokens[start:end]).strip()
            if piece:
                chunks.append(piece)
            start += step

        return chunks

    def _deduplicate(self, chunks: Sequence[Chunk]) -> List[Chunk]:
        unique: List[Chunk] = []
        seen_hashes = set()
        embeddings: List[Any] = []

        for chunk in chunks:
            norm = self._normalize_for_dedup(chunk.text)
            if not norm:
                continue

            fingerprint = hashlib.md5(norm.encode("utf-8")).hexdigest()
            if fingerprint in seen_hashes:
                continue

            emb = self.embedder.embed_text(norm)
            is_duplicate = False
            for prev_emb in embeddings:
                sim = self._cosine_similarity(emb, prev_emb)
                if sim >= self.dedup_similarity:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            seen_hashes.add(fingerprint)
            embeddings.append(emb)
            unique.append(chunk)

        return unique

    def _build_chunk(self, text: str, metadata: Dict[str, Any], parent_id: str, index: int) -> Chunk:
        text = self._clean_text(text)
        chunk_id = self._make_chunk_id(parent_id, index, text)

        full_metadata = dict(metadata)
        full_metadata["token_count"] = len(self.tokenizer.encode(text))
        full_metadata.setdefault("language", self._detect_language(text))
        full_metadata.setdefault("content_hash", hashlib.md5(text.encode("utf-8")).hexdigest())

        return Chunk(chunk_id=chunk_id, text=text, metadata=full_metadata)

    def _make_chunk_id(self, parent_id: str, index: int, text: str) -> str:
        import time
        raw = f"{parent_id}:{index}:{text[:120]}:{time.time_ns()}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
        return f"chunk_{digest}"

    def _normalize_for_dedup(self, text: str) -> str:
        text = self._clean_text(text).lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s%/-]", "", text)
        return text.strip()

    def _clean_text(self, text: str) -> str:
        text = text or ""
        text = text.replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _dynamic_similarity_threshold(self, similarities: Sequence[float]) -> float:
        if not similarities:
            return 0.75
        mean = sum(similarities) / len(similarities)
        variance = sum((x - mean) ** 2 for x in similarities) / len(similarities)
        std = math.sqrt(variance)
        return max(0.45, min(0.9, mean - 0.5 * std))

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        num = sum(x * y for x, y in zip(a, b))
        den_a = math.sqrt(sum(x * x for x in a))
        den_b = math.sqrt(sum(y * y for y in b))
        if den_a == 0 or den_b == 0:
            return 0.0
        return num / (den_a * den_b)

    def _detect_language(self, text: str) -> str:
        if re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", text.lower()):
            return "vi"
        return "unknown"