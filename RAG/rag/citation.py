from __future__ import annotations

import re
from typing import Dict, List


class CitationMapper:
    def __init__(self, metadatas: List[Dict]):
        self.metadatas = metadatas or []

    def replace(self, answer: str) -> str:
        """
        Nếu model sinh [1], [2]... thì map sang nguồn thật.
        Nếu không có placeholder thì giữ nguyên.
        """
        if not answer:
            return answer

        def repl(match):
            idx = int(match.group(1)) - 1
            if idx < 0 or idx >= len(self.metadatas):
                return match.group(0)

            meta = self.metadatas[idx]
            source = meta.get("source", "unknown")
            page = meta.get("page", "N/A")
            return f"[{source}, trang {page}]"

        return re.sub(r"\[(\d+)\]", repl, answer)

    def print_citations(self) -> None:
        if not self.metadatas:
            return

        print("\nNguồn tham chiếu:")
        for i, meta in enumerate(self.metadatas, start=1):
            source = meta.get("source", "unknown")
            page = meta.get("page", "N/A")
            section = meta.get("raw_section_heading") or meta.get("section_heading") or ""
            line = f"[{i}] {source}, trang {page}"
            if section:
                line += f" | section={section}"
            print(line)