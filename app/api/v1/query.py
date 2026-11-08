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
from app.schemas.query import QueryLogRead, QueryRequest, QueryResponse
from app.services.answering import GenerationError, answer_question
from app.services.embedder import EmbeddingError, EmbeddingService, get_embedder

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
    """A single-turn answer from the indexed documents, with its citations.
    The answering lives in `app/services/answering.py`, shared with the MCP
    server's `ask_documents`."""
    try:
        return await answer_question(db, embedder, payload)
    except EmbeddingError as exc:
        logger.exception("query_embedding_failed")
        raise upstream_error(exc, "Embedding the question") from exc
    except GenerationError as exc:
        logger.exception("query_generation_failed")
        raise upstream_error(exc, "Generation") from exc


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
