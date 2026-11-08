from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Conversation, SavedNote
from app.schemas.note import SavedNoteRead

router = APIRouter(prefix="/notes", tags=["notes"])


@router.get("", operation_id="list_notes", response_model=list[SavedNoteRead])
async def list_notes(
    limit: int = Query(default=50, ge=1, le=200),
    collection_id: int | None = None,
    conversation_id: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[SavedNoteRead]:
    # With each note's thread id, so a client can link the note to its thread.
    stmt = (
        select(SavedNote, Conversation.thread_id)
        .outerjoin(Conversation, SavedNote.conversation_id == Conversation.id)
        .order_by(SavedNote.created_at.desc())
        .limit(limit)
    )
    if collection_id is not None:
        stmt = stmt.where(SavedNote.collection_id == collection_id)
    if conversation_id is not None:
        stmt = stmt.where(SavedNote.conversation_id == conversation_id)
    rows = (await db.execute(stmt)).all()
    return [
        SavedNoteRead.model_validate(note).model_copy(update={"thread_id": thread_id})
        for note, thread_id in rows
    ]


@router.delete("/{note_id}", operation_id="delete_note", status_code=status.HTTP_204_NO_CONTENT)
async def delete_note(
    note_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    note = await db.get(SavedNote, note_id)
    if note is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    await db.delete(note)
    await db.commit()
