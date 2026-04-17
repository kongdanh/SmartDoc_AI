"""Pydantic request/response models for the KnowledgeDB API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query")
    domain: str = Field(..., min_length=1, description="Knowledge domain to search")
    community_level: int = Field(default=2, ge=0, le=10, description="Community hierarchy level")
    response_type: str = Field(
        default="Multiple Paragraphs",
        description="Response format: 'Multiple Paragraphs', 'Single Paragraph', 'Single Sentence', etc.",
    )


class GlobalQueryRequest(QueryRequest):
    dynamic_community_selection: bool = Field(
        default=False,
        description="Enable dynamic community selection for global search",
    )


class QueryResponse(BaseModel):
    response: str | None = None
    domain: str
    method: str
    error: str | None = None


class DirectSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query")
    domain: str = Field(..., min_length=1, description="Knowledge domain to search")
    top_k: int = Field(default=10, ge=1, le=50, description="Max results per category")


class DirectSearchResponse(BaseModel):
    domain: str
    method: str = "direct"
    entities: list[dict] = []
    relationships: list[dict] = []
    sources: list[dict] = []
    error: str | None = None


class DomainInfo(BaseModel):
    name: str
    ready: bool
    indexed_files: int = 0


class DomainListResponse(BaseModel):
    domains: list[DomainInfo]


class StatusResponse(BaseModel):
    server: str = "running"
    domains_total: int
    domains_ready: int
    indexing_status: dict[str, dict]


class ReindexResponse(BaseModel):
    message: str
    domains: list[str]


# --- Chat models ---


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    domain: str = Field(..., min_length=1, description="Knowledge domain for context retrieval")
    session_id: str | None = Field(default=None, description="Session ID for multi-turn conversation")
