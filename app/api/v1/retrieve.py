"""The chunks for a query, without an answer (audit R20).

What another agent needs from a knowledge base: the passages and where they
come from, to answer with its own model. The search is the one the chat and
`/query` use (hybrid retrieval, then the reranker and MMR when they are on);
only the generation is left out. It pays one embedding, so it counts against
the questions rate limit.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import upstream_error
from app.api.rate_limit import questions_limit
from app.config import settings
from app.db import get_db
from app.logging_config import get_logger
from app.models import Collection, Document
from app.schemas.retrieve import (
    RetrievedChunkRead,
    RetrieveMetadata,
    RetrieveRequest,
    RetrieveResponse,
)
from app.services.embedder import EmbeddingError, EmbeddingService, get_embedder
from app.services.retrieval_pipeline import retrieve as retrieve_pipeline

logger = get_logger(__name__)

router = APIRouter(tags=["retrieval"])


async def _scope(db: AsyncSession, payload: RetrieveRequest) -> int | None:
    """The collection to search: the document's when it names one. A
    collection or a document that does not exist is a 404."""
    if payload.document_id is not None:
        document = await db.get(Document, payload.document_id)
        if document is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Document {payload.document_id} not found"
            )
        if payload.collection_id not in (None, document.collection_id):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Document {document.id} is not in collection {payload.collection_id}",
            )
        return document.collection_id
    if payload.collection_id is not None:
        exists = await db.scalar(
            select(Collection.id).where(Collection.id == payload.collection_id)
        )
        if exists is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Collection {payload.collection_id} not found"
            )
    return payload.collection_id


@router.post(
    "/retrieve",
    operation_id="retrieve_chunks",
    response_model=RetrieveResponse,
    dependencies=[Depends(questions_limit)],
)
async def retrieve_chunks(
    payload: RetrieveRequest,
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder),
) -> RetrieveResponse:
    """The chunks that best match `query`, most relevant first, with their text
    and their document: no answer is generated. Scope it with `collection_id`,
    or with `document_id` for a single document."""
    start = time.perf_counter()
    collection_id = await _scope(db, payload)

    try:
        embedding = await embedder.embed(payload.query)
    except EmbeddingError as exc:
        logger.exception("retrieve_embedding_failed")
        raise upstream_error(exc, "Embedding the query") from exc

    chunks = await retrieve_pipeline(
        db=db,
        query_text=payload.query,
        query_embedding=embedding,
        collection_id=collection_id,
        top_k=payload.top_k,
        document_id=payload.document_id,
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
