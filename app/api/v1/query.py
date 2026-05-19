import time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import QueryLog
from app.schemas.pagination import Page
from app.schemas.query import (
    Citation,
    QueryLogRead,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
)
from app.services.embedder import EmbeddingError, EmbeddingService, get_embedder
from app.services.llm import LLMError, llm_service_for
from app.services.llm.pricing import estimate_cost_usd
from app.services.rag import RAG_SYSTEM_PROMPT, build_user_prompt
from app.services.retrieval_pipeline import retrieve as retrieve_pipeline

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder),
) -> QueryResponse:
    start = time.perf_counter()
    provider, model, temperature = llm_service_for("rag")

    try:
        question_embedding = await embedder.embed(payload.question)
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    chunks = await retrieve_pipeline(
        db=db,
        query_text=payload.question,
        query_embedding=question_embedding,
        collection_id=payload.collection_id,
        top_k=payload.top_k,
    )

    if not chunks:
        answer = "I don't have any indexed documents to answer this question."
        latency_ms = int((time.perf_counter() - start) * 1000)

        log = QueryLog(
            question=payload.question,
            answer=answer,
            collection_id=payload.collection_id,
            chunks_retrieved=0,
            model_used=model,
            tokens_input=0,
            tokens_output=0,
            latency_ms=latency_ms,
            session_id=payload.session_id,
        )
        db.add(log)
        await db.commit()

        return QueryResponse(
            answer=answer,
            citations=[],
            metadata=QueryMetadata(
                model=model,
                tokens_input=0,
                tokens_output=0,
                latency_ms=latency_ms,
                chunks_retrieved=0,
            ),
        )

    user_prompt = build_user_prompt(payload.question, chunks)

    try:
        llm_response = await provider.generate(
            system=RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            model=model,
            temperature=temperature,
            max_tokens=1024,
        )
    except LLMError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    latency_ms = int((time.perf_counter() - start) * 1000)

    log = QueryLog(
        question=payload.question,
        answer=llm_response.text,
        collection_id=payload.collection_id,
        chunks_retrieved=len(chunks),
        model_used=llm_response.model,
        tokens_input=llm_response.tokens_input,
        tokens_output=llm_response.tokens_output,
        latency_ms=latency_ms,
        cost_usd=estimate_cost_usd(
            llm_response.model, llm_response.tokens_input, llm_response.tokens_output
        ),
        session_id=payload.session_id,
    )
    db.add(log)
    await db.commit()

    return QueryResponse(
        answer=llm_response.text,
        citations=[
            Citation(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_title=c.document_title,
                chunk_index=c.chunk_index,
                relevance=c.relevance,
            )
            for c in chunks
        ],
        metadata=QueryMetadata(
            model=llm_response.model,
            tokens_input=llm_response.tokens_input,
            tokens_output=llm_response.tokens_output,
            latency_ms=latency_ms,
            chunks_retrieved=len(chunks),
        ),
    )


@router.get("/history", response_model=Page[QueryLogRead])
async def query_history(
    limit: int = 50,
    cursor: int | None = None,
    session_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> Page[QueryLogRead]:
    q = select(QueryLog).order_by(QueryLog.id.desc()).limit(limit + 1)
    if cursor is not None:
        q = q.where(QueryLog.id < cursor)
    if session_id:
        q = q.where(QueryLog.session_id == session_id)
    rows = list((await db.execute(q)).scalars().all())
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].id if has_more and items else None
    return Page[QueryLogRead](
        items=[QueryLogRead.model_validate(r) for r in items],
        next_cursor=next_cursor,
    )
