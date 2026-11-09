from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    collection_id: int | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    # query_logs.session_id is String(50); longer failed the commit with a 500
    # after the LLM call was already paid for.
    session_id: str | None = Field(default=None, max_length=50)


class Citation(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    # Score the citations are ordered by, 0-1 (see RetrievedChunk).
    relevance: float
    # Cosine similarity to the question; None if only full-text search found it.
    cosine: float | None = None


class QueryMetadata(BaseModel):
    model: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    chunks_retrieved: int
    cost_usd: float = 0.0


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
    cost_usd: float
    session_id: str | None
    agent_action: str | None
    # The turn's OpenTelemetry trace and the prompts that answered it
    # (app/telemetry.py, app/agent/prompt_version.py); null on rows from
    # before each column.
    trace_id: str | None = None
    prompt_version: str | None = None
    created_at: datetime
