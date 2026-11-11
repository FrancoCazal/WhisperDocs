"""Deleting and moving documents.

Behind the API's delete and bulk routes and the MCP server's
`delete_document` and `move_document` (audit M-1, BE-36), so both change the
knowledge base alike. Neither function commits: the caller does, so the MCP
server commits its audit row with the change.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document


async def delete_documents(db: AsyncSession, document_ids: Sequence[int]) -> set[int]:
    """Delete the documents in one statement; the foreign key removes their
    chunks (audit R11). Returns the ids that existed."""
    deleted = await db.scalars(
        delete(Document).where(Document.id.in_(list(document_ids))).returning(Document.id)
    )
    return set(deleted.all())


class MoveConflictError(Exception):
    """A document with the same text reached the target collection while
    moving. Nothing was moved."""

    def __init__(self) -> None:
        super().__init__(
            "A document with the same text reached the target collection while moving; "
            "nothing was moved. Try again."
        )


@dataclass
class MoveResult:
    # The requested documents now in the target, those already there included.
    moved: set[int] = field(default_factory=set)
    # Why a document stayed where it was, by its id.
    errors: dict[int, str] = field(default_factory=dict)


async def move_documents(
    db: AsyncSession, document_ids: Sequence[int], target_collection_id: int
) -> MoveResult:
    """Reassign the documents to the target collection, in one statement.

    A collection holds each document once (audit R12): one whose text the
    target already holds, or that another moved document brings first,
    stays where it is, with the reason in `errors`. The target must exist.
    Raises MoveConflictError, after a rollback, when an upload of the same
    text reaches the target in between.
    """
    ids = list(document_ids)
    rows = (
        await db.execute(
            select(Document.id, Document.content_hash, Document.collection_id)
            .where(Document.id.in_(ids))
            .order_by(Document.id)
        )
    ).all()
    held = dict(
        (
            await db.execute(
                select(Document.content_hash, Document.title).where(
                    Document.collection_id == target_collection_id,
                    Document.content_hash.is_not(None),
                    Document.id.not_in(ids),
                )
            )
        ).all()
    )
    errors: dict[int, str] = {}
    movable: list[int] = []
    # Who keeps each text among the requested documents.
    claimed: dict[str, int] = {}
    # The documents already in the target come first: they stay, and claim their text.
    for document_id, digest, collection_id in sorted(
        rows, key=lambda row: (row.collection_id != target_collection_id, row.id)
    ):
        if digest is not None and digest in held:
            errors[document_id] = (
                f'The target collection already holds this document, as "{held[digest]}".'
            )
            continue
        if digest is not None and digest in claimed:
            errors[document_id] = (
                f"Document {claimed[digest]}, also in this request, has the same text."
            )
            continue
        if digest is not None:
            claimed[digest] = document_id
        if collection_id != target_collection_id:
            movable.append(document_id)

    if movable:
        try:
            await db.execute(
                update(Document)
                .where(Document.id.in_(movable))
                .values(collection_id=target_collection_id)
            )
        except IntegrityError as exc:
            # An upload of the same text reached the target in between.
            await db.rollback()
            raise MoveConflictError() from exc
    moved = {document_id for document_id, _digest, _collection in rows} - set(errors)
    return MoveResult(moved=moved, errors=errors)
