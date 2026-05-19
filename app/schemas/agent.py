from pydantic import UUID4, BaseModel, Field


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    thread_id: UUID4 | None = None
    collection_id: int | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class AgentCitation(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    relevance: float
    text_preview: str | None = None


class AgentTurnMetadata(BaseModel):
    model: str | None = None
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: int = 0
    chunks_retrieved: int = 0


class AgentChatResponse(BaseModel):
    thread_id: str
    action: str | None
    answer: str
    citations: list[AgentCitation]
    metadata: AgentTurnMetadata
