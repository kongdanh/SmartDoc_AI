from __future__ import annotations

import re


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ")
    text = text.replace("\r", "\n")

    # normalize spaces but preserve useful line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # remove isolated page numbers
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    return text.strip()