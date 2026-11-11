"""The audit log of the MCP server's writes (BE-36, audit M-2).

Every call to a tool that writes leaves one row (`app/models/audit_log.py`),
however it ended:

  done       the document was added, deleted or moved
  declined   the user said no when asked to confirm
  cancelled  the user dismissed the question
  refused    the server would not: a read-only key, a collection out of
             reach, a client that cannot ask its user
  failed     it was tried and did not work: a duplicate, a file the
             extractor refused, an embedding error

A write that happens commits its row in the same transaction as the change,
so neither lands without the other (`row`, added by the caller before its
commit). The others commit their row on its own (`record`).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.logging_config import get_logger
from app.models import AuditLog
from app.schemas.audit import AuditLogRead
from app.schemas.pagination import Page

logger = get_logger(__name__)

Outcome = Literal["done", "declined", "cancelled", "refused", "failed"]
Sessions = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def row(
    *,
    actor: str,
    channel: str,
    action: str,
    outcome: Outcome,
    collection_id: int | None = None,
    document_id: int | None = None,
    **detail: Any,
) -> AuditLog:
    return AuditLog(
        actor=actor,
        channel=channel,
        action=action,
        outcome=outcome,
        collection_id=collection_id,
        document_id=document_id,
        detail={k: v for k, v in detail.items() if v is not None},
    )


async def record(sessions: Sessions, **fields: Any) -> None:
    """Commit one row on its own, for a write that did not happen. A failure
    to record is logged, not raised: the caller is already answering with
    why the write did not happen."""
    entry = row(**fields)
    try:
        async with sessions() as db:
            db.add(entry)
            await db.commit()
    except Exception:
        logger.exception(
            "audit_record_failed", action=entry.action, outcome=entry.outcome, actor=entry.actor
        )


async def list_entries(db: AsyncSession, *, limit: int, cursor: int | None) -> Page[AuditLogRead]:
    """A page of the log, newest first (see `app/schemas/pagination.py`)."""
    query = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit + 1)
    if cursor is not None:
        query = query.where(AuditLog.id < cursor)
    rows = list((await db.execute(query)).scalars().all())
    has_more = len(rows) > limit
    items = rows[:limit]
    total = await db.scalar(select(func.count(AuditLog.id))) if cursor is None else None
    return Page[AuditLogRead](
        items=[AuditLogRead.model_validate(r) for r in items],
        next_cursor=items[-1].id if has_more and items else None,
        total=total,
    )
