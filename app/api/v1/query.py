import time
from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import upstream_error
from app.api.rate_limit import questions_limit
from app.db import contains_text, get_db
from app.logging_config import get_logger
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
from app.services.llm import llm_service_for
from app.services.llm.pricing import estimate_cost_usd
from app.services.rag import RAG_SYSTEM_PROMPT, build_user_prompt
from app.services.retrieval_pipeline import retrieve as retrieve_pipeline

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    operation_id="answer_question",
    response_model=QueryResponse,
    dependencies=[Depends(questions_limit)],
)
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
        logger.exception("query_embedding_failed")
        raise upstream_error(exc, "Embedding the question") from exc

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

    # The citations index into the chunks that fit the context budget.
    user_prompt, included = build_user_prompt(payload.question, chunks)

    try:
        llm_response = await provider.generate(
            system=RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            model=model,
            temperature=temperature,
            max_tokens=1024,
        )
    except Exception as exc:
        # The providers raise their SDK's exceptions, not LLMError: catching
        # that alone let every model failure through as a bare 500.
        logger.exception("query_generation_failed", model=model)
        raise upstream_error(exc, "Generation") from exc

    latency_ms = int((time.perf_counter() - start) * 1000)
    cost_usd = estimate_cost_usd(
        llm_response.model, llm_response.tokens_input, llm_response.tokens_output
    )

    log = QueryLog(
        question=payload.question,
        answer=llm_response.text,
        collection_id=payload.collection_id,
        chunks_retrieved=len(chunks),
        model_used=llm_response.model,
        tokens_input=llm_response.tokens_input,
        tokens_output=llm_response.tokens_output,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
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
                cosine=c.cosine,
            )
            for c in included
        ],
        metadata=QueryMetadata(
            model=llm_response.model,
            tokens_input=llm_response.tokens_input,
            tokens_output=llm_response.tokens_output,
            latency_ms=latency_ms,
            chunks_retrieved=len(chunks),
            cost_usd=cost_usd,
        ),
    )


@router.get("/history", operation_id="list_query_history", response_model=Page[QueryLogRead])
async def query_history(
    limit: int = Query(default=50, ge=1, le=200),
    cursor: int | None = None,
    session_id: str | None = Query(default=None, max_length=50),
    collection_id: int | None = None,
    source: Literal["chat", "query"] | None = Query(
        default=None, description="chat: the agent's answers; query: POST /query's"
    ),
    q: str | None = Query(default=None, max_length=200, description="Text the question contains"),
    db: AsyncSession = Depends(get_db),
) -> Page[QueryLogRead]:
    filters = []
    # The agent logs its node as agent_action; /query logs none.
    if source == "chat":
        filters.append(QueryLog.agent_action.is_not(None))
    elif source == "query":
        filters.append(QueryLog.agent_action.is_(None))
    if q and q.strip():
        filters.append(contains_text(QueryLog.question, q.strip()))
    if session_id:
        filters.append(QueryLog.session_id == session_id)
    if collection_id is not None:
        filters.append(QueryLog.collection_id == collection_id)
    q = select(QueryLog).where(*filters).order_by(QueryLog.id.desc()).limit(limit + 1)
    if cursor is not None:
        q = q.where(QueryLog.id < cursor)
    rows = list((await db.execute(q)).scalars().all())
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].id if has_more and items else None
    total = None
    if cursor is None:
        total = await db.scalar(select(func.count(QueryLog.id)).where(*filters))
    return Page[QueryLogRead](
        items=[QueryLogRead.model_validate(r) for r in items],
        next_cursor=next_cursor,
        total=total,
    )
