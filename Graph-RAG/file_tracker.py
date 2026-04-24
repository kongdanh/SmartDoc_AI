"""
SmartDoc AI — File Registry.

Tracks which files have been indexed per domain.
Persists to a JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class FileRegistry:
    """Simple JSON-backed registry of indexed files."""

    def __init__(self, path: str | Path = "indexes/file_registry.json") -> None:
        self.path = Path(path)
        self._data: Dict[str, Any] = self._load()

    # ── persistence ─────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── domain-level operations ─────────────────────────────────

    def is_indexed(self, domain: str, filename: str) -> bool:
        return filename in self._data.get(domain, {}).get("files", {})

    def mark_indexed(self, domain: str, filename: str, checksum: str = "") -> None:
        if domain not in self._data:
            self._data[domain] = {"files": {}, "indexed_files": 0}
        self._data[domain]["files"][filename] = {"checksum": checksum}
        self._data[domain]["indexed_files"] = len(self._data[domain]["files"])
        self._save()

    def clear_domain(self, domain: str) -> None:
        self._data.pop(domain, None)
        self._save()

    def get_stats(self) -> Dict[str, Any]:
        return {
            domain: {"indexed_files": info.get("indexed_files", 0)}
            for domain, info in self._data.items()
        }

    def get_domain_files(self, domain: str) -> list[str]:
        return list(self._data.get(domain, {}).get("files", {}).keys())
