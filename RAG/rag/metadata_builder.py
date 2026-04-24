from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional


class MetadataBuilder:
    def build_metadata(
        self,
        filename: str,
        page_number: int,
        chunk_index: Optional[int] = None,
        text: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        filename_only = Path(filename).name
        text = text or ""

        metadata: Dict[str, Any] = {
            "filename": filename_only,
            "source": filename_only,
            "page": int(page_number),
            "source_label": f"{filename_only}, trang {page_number}",
            "raw_title": "",
            "raw_section_heading": "",
            "content_hash": hashlib.md5(text.encode("utf-8")).hexdigest() if text else "",
        }

        if chunk_index is not None:
            metadata["chunk_index"] = int(chunk_index)

        if extra:
            metadata.update(extra)

        return metadata