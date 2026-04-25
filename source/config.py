"""
SmartDoc AI — Application Settings.

Reads .env and provides typed settings for the entire application.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent


class _Settings:
    """Central configuration object — created once, used everywhere."""

    def __init__(self) -> None:
        # ── Paths ───────────────────────────────────────────────
        self.data_path: Path = _ROOT / os.getenv("DATA_DIR", "data")
        self.index_path: Path = _ROOT / os.getenv("INDEX_DIR", "indexes")
        self.upload_path: Path = _ROOT / "resources" / "uploads"

        # ── Server ──────────────────────────────────────────────
        self.server_port: int = int(os.getenv("SERVER_PORT", "8001"))

        # ── LLM (OpenRouter — used by GraphRAG and Standard RAG) 
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "openrouter")
        self.llm_base_url: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        self.llm_model: str = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        self.llm_api_key: str = os.getenv("LLM_API_KEY", "")
        self.llm_rpm: int = int(os.getenv("LLM_RPM", "5"))

        # ── Embedding ───────────────────────────────────────
        self.embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

        # ── Indexing ────────────────────────────────────────────
        self.indexing_method: str = os.getenv("INDEXING_METHOD", "standard")

        # ── Ensure directories exist ───────────────────────────
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.upload_path.mkdir(parents=True, exist_ok=True)
        (self.upload_path / "pdf").mkdir(exist_ok=True)
        (self.upload_path / "docx").mkdir(exist_ok=True)
        (self.upload_path / "txt").mkdir(exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"Settings(data={self.data_path}, index={self.index_path}, "
            f"upload={self.upload_path}, port={self.server_port})"
        )


settings = _Settings()
