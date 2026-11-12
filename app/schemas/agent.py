from pydantic import UUID4, BaseModel, Field


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    thread_id: UUID4 | None = None
    collection_id: int | None = None
    # One document to search in place of the collection. Like collection_id,
    # a turn that names it changes the thread's, and null goes back to the
    # collection (see apply_turn_scope).
    document_id: int | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class AgentCitation(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    # Score the citations are ordered by, 0-1 (see RetrievedChunk).
    relevance: float
    # Cosine similarity to the question; None if only full-text search found it.
    cosine: float | None = None
    text_preview: str | None = None
    # Where the chunk sits (app/models/chunk.py); missing from citations
    # stored before.
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None


class AgentTurnMetadata(BaseModel):
    model: str | None = None
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: int = 0
    chunks_retrieved: int = 0
    # The answer's model call, priced by the backend (`query_logs.cost_usd`).
    cost_usd: float = 0.0


class TurnErrorRead(BaseModel):
    """A call the turn made that failed or fell back (see
    app/agent/utils/turn_errors.py for the steps)."""

    # summarize, classifier, query_rewrite, embedding, retrieval, generation,
    # clarification, citation_check
    step: str
    # billing, auth, rate_limit, timeout, unavailable, invalid_request,
    # bad_response, unknown (app/services/llm/errors.py)
    kind: str
    model: str | None = None


class AgentChatResponse(BaseModel):
    thread_id: str
    action: str | None
    answer: str
    citations: list[AgentCitation]
    metadata: AgentTurnMetadata
    # The answer reports a failure ("I had a problem…"): it cites nothing and
    # later prompts leave it out (audit N4).
    failed: bool = False
    # The turn's failed calls. A turn can answer with some: when the
    # classifier fails it searches the documents instead.
    errors: list[TurnErrorRead] = []
