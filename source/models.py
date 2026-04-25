"""
SmartDoc AI — Pydantic request / response models for FastAPI.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# ── Domain ──────────────────────────────────────────────────────

class DomainInfo(BaseModel):
    name: str
    ready: bool = False
    indexed_files: int = 0


class DomainListResponse(BaseModel):
    domains: list[DomainInfo] = []


# ── Query ───────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    domain: str
    query: str
    community_level: int = 2
    response_type: str = "Multiple Paragraphs"


class GlobalQueryRequest(QueryRequest):
    dynamic_community_selection: bool = False


class QueryResponse(BaseModel):
    response: str = ""
    method: str = ""
    error: Optional[str] = None


class DirectSearchRequest(BaseModel):
    domain: str
    query: str
    top_k: int = 10


class DirectSearchResponse(BaseModel):
    entities: list[dict] = []
    relationships: list[dict] = []
    sources: list[dict] = []
    error: Optional[str] = None


# ── Chat ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    domain: str
    message: str
    session_id: Optional[str] = None


class CompareRequest(BaseModel):
    query: str
    domain: str


# ── Status ──────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    domains_total: int = 0
    domains_ready: int = 0
    indexing_status: dict = {}


class ReindexResponse(BaseModel):
    message: str = ""
    domains: list[str] = []
