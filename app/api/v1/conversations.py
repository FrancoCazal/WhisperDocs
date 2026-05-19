import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Conversation, SavedNote
from app.schemas.conversation import (
    ConversationMessage,
    ConversationRead,
    ConversationUpdate,
)
from app.schemas.note import SavedNoteCreate, SavedNoteRead
from app.schemas.pagination import Page

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=Page[ConversationRead])
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    cursor: int | None = Query(default=None, description="Last id from previous page"),
    user_number: str | None = None,
    is_saved: bool | None = None,
    channel: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> Page[ConversationRead]:
    stmt = select(Conversation).order_by(Conversation.id.desc()).limit(limit + 1)
    if cursor is not None:
        stmt = stmt.where(Conversation.id < cursor)
    if user_number is not None:
        stmt = stmt.where(Conversation.user_number == user_number)
    if is_saved is not None:
        stmt = stmt.where(Conversation.is_saved == is_saved)
    if channel is not None:
        stmt = stmt.where(Conversation.channel == channel)
    rows = list((await db.execute(stmt)).scalars().all())
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].id if has_more and items else None
    return Page[ConversationRead](
        items=[ConversationRead.model_validate(c) for c in items],
        next_cursor=next_cursor,
    )


async def _get_conversation_or_404(db: AsyncSession, thread_id: str) -> Conversation:
    conv = await db.scalar(select(Conversation).where(Conversation.thread_id == thread_id))
    if conv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conv


@router.get("/{thread_id}/messages", response_model=list[ConversationMessage])
async def get_conversation_messages(
    thread_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> list[ConversationMessage]:
    await _get_conversation_or_404(db, thread_id)

    agent = request.app.state.agent
    snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    messages = (snapshot.values or {}).get("messages", []) if snapshot else []

    out: list[ConversationMessage] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, SystemMessage):
            role = "system"
        else:
            continue
        content = m.content if isinstance(m.content, str) else str(m.content)
        kwargs = getattr(m, "additional_kwargs", None) or {}
        out.append(
            ConversationMessage(
                role=role,
                content=content,
                timestamp=kwargs.get("timestamp"),
                citations=kwargs.get("citations"),
                metadata=kwargs.get("metadata"),
            )
        )
    return out


@router.patch("/{thread_id}", response_model=ConversationRead)
async def patch_conversation(
    thread_id: str,
    payload: ConversationUpdate,
    db: AsyncSession = Depends(get_db),
) -> Conversation:
    conv = await _get_conversation_or_404(db, thread_id)
    conv.is_saved = payload.is_saved
    await db.commit()
    await db.refresh(conv)
    return conv


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    thread_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> None:
    conv = await _get_conversation_or_404(db, thread_id)
    await db.delete(conv)

    agent = request.app.state.agent
    try:
        await agent.checkpointer.adelete_thread(thread_id)
    except Exception:
        logger.exception("checkpointer_delete_failed thread_id=%s", thread_id)
        # Don't block the DB deletion — log and continue.

    await db.commit()


@router.post(
    "/{thread_id}/notes",
    response_model=SavedNoteRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_note_in_conversation(
    thread_id: str,
    payload: SavedNoteCreate,
    db: AsyncSession = Depends(get_db),
) -> SavedNote:
    conv = await _get_conversation_or_404(db, thread_id)
    note = SavedNote(
        conversation_id=conv.id,
        collection_id=conv.collection_id,
        source_message_content=payload.content,
        user_note=payload.user_note,
    )
    db.add(note)
    await db.commit()
    await db.refresh(note)
    return note
