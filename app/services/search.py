"""The chunks for a query, without an answer.

Behind `POST /retrieve` (audit R20) and the MCP server's `search_documents`
(audit M-1), so both return the same chunks for the same request. The search
is the one the chat and `/query` use (hybrid retrieval, then the reranker and
MMR when they are on); only the generation is left out.
"""

from __future__ import annotations

import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Collection, Document
from app.schemas.retrieve import (
    RetrievedChunkRead,
    RetrieveMetadata,
    RetrieveRequest,
    RetrieveResponse,
)
from app.services.embeddings import EmbeddingProvider
from app.services.retrieval_pipeline import retrieve as retrieve_pipeline


class ScopeError(Exception):
    """The scope a search names does not hold."""


class ScopeNotFound(ScopeError):
    """The document or the collection does not exist."""


class OutOfScope(ScopeError):
    """The document is not in the collection the search names."""


async def resolve_scope(
    db: AsyncSession, *, collection_id: int | None, document_id: int | None
) -> int | None:
    """The collection to search: the document's when it names one."""
    if document_id is not None:
        document = await db.get(Document, document_id)
        if document is None:
            raise ScopeNotFound(f"Document {document_id} not found")
        if collection_id not in (None, document.collection_id):
            raise OutOfScope(f"Document {document.id} is not in collection {collection_id}")
        return document.collection_id
    if collection_id is not None:
        exists = await db.scalar(select(Collection.id).where(Collection.id == collection_id))
        if exists is None:
            raise ScopeNotFound(f"Collection {collection_id} not found")
    return collection_id


async def search(
    db: AsyncSession, embedder: EmbeddingProvider, request: RetrieveRequest
) -> RetrieveResponse:
    """The chunks that best match the query, most relevant first.

    The scope is checked before the embedding is paid for. Raises
    `ScopeNotFound`, `OutOfScope`, or the embedder's `EmbeddingError`.
    """
    start = time.perf_counter()
    collection_id = await resolve_scope(
        db, collection_id=request.collection_id, document_id=request.document_id
    )
    embedding = await embedder.embed(request.query)
    chunks = await retrieve_pipeline(
        db=db,
        query_text=request.query,
        query_embedding=embedding,
        collection_id=collection_id,
        top_k=request.top_k,
        document_id=request.document_id,
    )
    return RetrieveResponse(
        chunks=[
            RetrievedChunkRead(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_title=c.document_title,
                chunk_index=c.chunk_index,
                text=c.text,
                relevance=c.relevance,
                cosine=c.cosine,
                section=c.section,
                page_start=c.page_start,
                page_end=c.page_end,
            )
            for c in chunks
        ],
        metadata=RetrieveMetadata(
            embedding_model=settings.embedding_model,
            reranker=settings.reranker_model if settings.reranker_enabled else None,
            mmr=settings.mmr_enabled,
            latency_ms=int((time.perf_counter() - start) * 1000),
        ),
    )
