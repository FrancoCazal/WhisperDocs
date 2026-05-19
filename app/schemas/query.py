from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    collection_id: int | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = None


class Citation(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    relevance: float


class QueryMetadata(BaseModel):
    model: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    chunks_retrieved: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    metadata: QueryMetadata


class QueryLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    question: str
    answer: str
    collection_id: int | None
    chunks_retrieved: int
    model_used: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    session_id: str | None
    agent_action: str | None
    created_at: datetime
