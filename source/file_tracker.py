"""Hash-based file registry for tracking indexed files per domain."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileRegistry:
    """Tracks file hashes to detect new or modified files across domains.

    Registry format (JSON):
    {
        "domain_a": {
            "relative/path/to/file.txt": "sha256_hash",
            ...
        },
        ...
    }
    """

    def __init__(self, registry_path: Path):
        self._path = registry_path
        self._data: dict[str, dict[str, str]] = self._load()

    def _load(self) -> dict[str, dict[str, str]]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupted registry file, starting fresh: %s", self._path)
        return {}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_new_or_changed_files(
        self, domain: str, file_paths: list[Path], base_dir: Path
    ) -> list[Path]:
        """Return files that are new or have changed since last indexing."""
        domain_hashes = self._data.get(domain, {})
        changed: list[Path] = []

        for fpath in file_paths:
            rel = str(fpath.relative_to(base_dir))
            current_hash = self._hash_file(fpath)
            if current_hash is None:
                continue
            if domain_hashes.get(rel) != current_hash:
                changed.append(fpath)

        return changed

    def mark_indexed(self, domain: str, file_paths: list[Path], base_dir: Path) -> None:
        """Record hashes for successfully indexed files."""
        if domain not in self._data:
            self._data[domain] = {}

        for fpath in file_paths:
            rel = str(fpath.relative_to(base_dir))
            h = self._hash_file(fpath)
            if h is not None:
                self._data[domain][rel] = h

        self.save()

    def domain_needs_reindex(
        self, domain: str, file_paths: list[Path], base_dir: Path
    ) -> bool:
        """True if any file in the domain is new or changed."""
        return len(self.get_new_or_changed_files(domain, file_paths, base_dir)) > 0

    def clear_domain(self, domain: str) -> None:
        """Remove all tracking data for a domain (used before full re-index)."""
        self._data.pop(domain, None)
        self.save()

    def list_domains(self) -> list[str]:
        return list(self._data.keys())

    def get_stats(self) -> dict[str, Any]:
        return {
            domain: {"indexed_files": len(files)}
            for domain, files in self._data.items()
        }

    @staticmethod
    def _hash_file(file_path: Path) -> str | None:
        try:
            sha = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
            return sha.hexdigest()
        except OSError:
            logger.warning("Cannot hash file: %s", file_path)
            return None
