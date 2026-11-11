from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.schemas.audit import AuditLogRead
from app.schemas.model_status import ModelStatusRead
from app.schemas.pagination import Page
from app.services import audit
from app.services.model_status import model_status

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/settings", operation_id="get_settings")
async def get_settings() -> dict:
    """Read-only snapshot of agent settings. Useful for UI debugging.

    To change values: set AGENT_* env vars and restart. Runtime mutation
    is intentionally not exposed (would invalidate compiled graph).
    """
    a = settings.agent
    return {
        "models": {
            "classifier": {"model": a.model_classifier, "temperature": a.temp_classifier},
            "query_rewrite": {"model": a.model_query_rewrite, "temperature": a.temp_query_rewrite},
            "rag_generation": {
                "model": a.model_rag_generation,
                "temperature": a.temp_rag_generation,
            },
            "clarification": {"model": a.model_clarification, "temperature": a.temp_clarification},
            "answer_direct": {"model": a.model_answer_direct, "temperature": a.temp_answer_direct},
            "summarize": {"model": a.model_summarize, "temperature": a.temp_summarize},
            "eval_judge": {"model": a.model_eval_judge, "temperature": a.temp_eval_judge},
            "extraction": {"model": a.model_extraction, "temperature": a.temp_extraction},
        },
        "retrieval": {
            "top_k": a.retrieval_top_k,
        },
        "compaction": {
            "token_threshold": a.compaction_token_threshold,
            "count_threshold": a.compaction_count_threshold,
            "k_recent": a.k_recent,
        },
        "embedding": {
            "model": settings.embedding_model,
            "dimension": settings.embedding_dimension,
        },
    }


@router.get("/model-status", operation_id="get_model_status", response_model=ModelStatusRead)
async def get_model_status() -> dict:
    """What the last calls of each role did: the classifier, the query rewrite,
    ..., the embedder. Read from the record real calls leave; nothing is called
    here, so it costs nothing to poll. A role with no call since the API
    started is "unknown"."""
    return {"since": model_status.since, "roles": model_status.snapshot()}


@router.get("/audit-log", operation_id="list_audit_log", response_model=Page[AuditLogRead])
async def list_audit_log(
    limit: int = Query(default=50, ge=1, le=200),
    cursor: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> Page[AuditLogRead]:
    """The calls that wrote through the MCP server, newest first: who, which
    tool, on which document and collection, and how each ended (done,
    declined, cancelled, refused, failed)."""
    return await audit.list_entries(db, limit=limit, cursor=cursor)
