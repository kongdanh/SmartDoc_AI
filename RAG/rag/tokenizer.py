from __future__ import annotations

from typing import List


class Tokenizer:
    """
    Tokenizer đơn giản đủ dùng cho chunk sizing nội bộ.
    Không cần exact tokenizer model-level.
    """

    def encode(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        return text.split()

    def decode(self, tokens: List[str]) -> str:
        return " ".join(tokens or [])