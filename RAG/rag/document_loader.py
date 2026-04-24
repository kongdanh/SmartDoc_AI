from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import fitz


TABLE_SPLIT_PATTERN = re.compile(r"\s{2,}|\t+|\|")
LIST_PATTERN = re.compile(r"^(\-|\•|\*|\d+[\.\)])\s+")
UPPER_HEADING_PATTERN = re.compile(r"^[A-ZÀ-Ỹ0-9\s\-\(\)\/,&%\.]+$")


def _clean_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def _looks_like_heading(line: str) -> bool:
    line = _clean_line(line)
    if not line:
        return False
    if len(line) > 140:
        return False

    if UPPER_HEADING_PATTERN.match(line) and len(line.split()) <= 14:
        return True

    if len(line.split()) <= 10 and not line.endswith((".", ";")):
        title_case_ratio = sum(1 for w in line.split() if w[:1].isupper()) / max(len(line.split()), 1)
        if title_case_ratio >= 0.6:
            return True

    return False


def _looks_like_list_item(line: str) -> bool:
    return bool(LIST_PATTERN.match(_clean_line(line)))


def _looks_like_table_row(line: str) -> bool:
    line = _clean_line(line)
    if not line:
        return False

    if TABLE_SPLIT_PATTERN.search(line):
        return True

    tokens = line.split()
    digit_ratio = sum(any(ch.isdigit() for ch in tok) for tok in tokens) / max(len(tokens), 1)
    return len(tokens) >= 4 and digit_ratio >= 0.3


def _extract_page_title(lines: List[str]) -> str:
    for line in lines[:8]:
        line = _clean_line(line)
        if _looks_like_heading(line):
            return line
    return ""


def _finalize_block(blocks: List[Dict], current_type: Optional[str], current_lines: List[str]) -> None:
    if not current_lines:
        return

    text = "\n".join(current_lines).strip()
    if not text:
        return

    blocks.append(
        {
            "type": current_type or "text",
            "text": text,
            "bbox": [0, len(blocks) * 100, 1000, len(blocks) * 100 + 80],
            "column": 0,
        }
    )


def _lines_to_blocks(lines: List[str]) -> List[Dict]:
    blocks: List[Dict] = []
    current_type: Optional[str] = None
    current_lines: List[str] = []

    for raw_line in lines:
        line = _clean_line(raw_line)

        if not line:
            _finalize_block(blocks, current_type, current_lines)
            current_type = None
            current_lines = []
            continue

        if _looks_like_heading(line):
            line_type = "heading"
        elif _looks_like_table_row(line):
            line_type = "table"
        elif _looks_like_list_item(line):
            line_type = "list"
        else:
            line_type = "text"

        if current_type is None:
            current_type = line_type
            current_lines = [line]
            continue

        if line_type != current_type:
            _finalize_block(blocks, current_type, current_lines)
            current_type = line_type
            current_lines = [line]
            continue

        current_lines.append(line)

    _finalize_block(blocks, current_type, current_lines)
    return blocks


def load_pdf(file_path: str) -> List[Dict]:
    doc = fitz.open(file_path)
    file_name = Path(file_path).name

    pages: List[Dict] = []

    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        text = text.replace("\r", "\n").strip()
        if not text:
            continue

        lines = text.split("\n")
        title = _extract_page_title(lines)
        blocks = _lines_to_blocks(lines)

        pages.append(
            {
                "page": i + 1,
                "title": title,
                "metadata": {
                    "filename": file_name,
                    "document_type": "pdf",
                    "loader": "pymupdf",
                    "raw_page_title": title,
                },
                "blocks": blocks,
            }
        )

    doc.close()
    return pages