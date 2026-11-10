"""What the knowledge base holds: its collections and its documents.

Behind the collection and document lists of the API and the MCP server's
`list_collections` and `list_documents` (audit M-1).
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.models import Collection, Document
from app.schemas.collection import CollectionRead
from app.schemas.document import DocumentRead
from app.schemas.pagination import Page


def _with_totals() -> Select:
    """Collections with their document, chunk and token totals."""
    totals = (
        select(
            Document.collection_id,
            func.count(Document.id).label("documents"),
            func.sum(Document.chunk_count).label("chunks"),
            func.sum(Document.total_tokens).label("tokens"),
        )
        .group_by(Document.collection_id)
        .subquery()
    )
    return select(
        Collection,
        func.coalesce(totals.c.documents, 0),
        func.coalesce(totals.c.chunks, 0),
        func.coalesce(totals.c.tokens, 0),
    ).outerjoin(totals, totals.c.collection_id == Collection.id)


def _read(row) -> CollectionRead:
    collection, documents, chunks, tokens = row
    return CollectionRead.model_validate(collection).model_copy(
        update={
            "document_count": int(documents),
            "chunk_count": int(chunks),
            "total_tokens": int(tokens),
        }
    )


async def list_collections(db: AsyncSession) -> list[CollectionRead]:
    """Every collection with its totals, newest first."""
    result = await db.execute(_with_totals().order_by(Collection.created_at.desc()))
    return [_read(row) for row in result.all()]


async def get_collection(db: AsyncSession, collection_id: int) -> CollectionRead | None:
    row = (await db.execute(_with_totals().where(Collection.id == collection_id))).first()
    return None if row is None else _read(row)


async def list_documents(
    db: AsyncSession,
    *,
    collection_id: int | None,
    limit: int,
    cursor: int | None,
    collection_ids: Iterable[int] | None = None,
) -> Page[DocumentRead]:
    """A page of documents, newest first. See `app/schemas/pagination.py`.

    Without their text, which a `DocumentRead` does not carry (audit R10).
    `collection_ids` keeps the page within those collections (an MCP key's).
    """
    filters = [] if collection_id is None else [Document.collection_id == collection_id]
    if collection_ids is not None:
        filters.append(Document.collection_id.in_(list(collection_ids)))
    query = (
        select(Document)
        .options(defer(Document.raw_text))
        .where(*filters)
        .order_by(Document.id.desc())
        .limit(limit + 1)
    )
    if cursor is not None:
        query = query.where(Document.id < cursor)
    rows = list((await db.execute(query)).scalars().all())
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].id if has_more and items else None
    total = None
    if cursor is None:
        total = await db.scalar(select(func.count(Document.id)).where(*filters))
    return Page[DocumentRead](
        items=[DocumentRead.model_validate(d) for d in items],
        next_cursor=next_cursor,
        total=total,
    )
