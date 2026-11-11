from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=2000,
        description="What to search for: a question, or keywords.",
    )
    collection_id: int | None = Field(
        default=None, description="Search only this collection. Null searches every collection."
    )
    document_id: int | None = Field(
        default=None,
        description="Search only this document. Its collection is implied; a different "
        "collection_id is a 400.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="How many chunks to return.")


class RetrievedChunkRead(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int = Field(description="The chunk's position in its document, from 0.")
    text: str
    relevance: float = Field(
        description="The score the chunks are ordered by (fused rank, or the reranker's), "
        "0-1. Relative to the other chunks of the same search: not a quality measure."
    )
    cosine: float | None = Field(
        description="Cosine similarity to the query. Null when only full-text search found "
        "the chunk."
    )
    section: str | None = Field(
        default=None,
        description="The headings the chunk is under, from the top level down, joined by "
        '" › ". Null when the document has none there.',
    )
    page_start: int | None = Field(
        default=None, description="The first page the chunk comes from, from 1; PDFs only."
    )
    page_end: int | None = Field(
        default=None, description="The last page the chunk comes from; PDFs only."
    )


class RetrieveMetadata(BaseModel):
    embedding_model: str
    reranker: str | None = Field(description="The cross-encoder that reordered the chunks, if any.")
    mmr: bool = Field(description="Whether MMR diversified the documents the chunks come from.")
    latency_ms: int


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunkRead]
    metadata: RetrieveMetadata
