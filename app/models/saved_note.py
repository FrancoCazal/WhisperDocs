from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class SavedNote(Base):
    __tablename__ = "saved_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # SET NULL: note survives deletion of its conversation. A note is
    # user-captured content — deleting the conversation must not destroy it.
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    )
    collection_id: Mapped[int | None] = mapped_column(
        ForeignKey("collections.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_message_content: Mapped[str] = mapped_column(Text, nullable=False)
    user_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Named as alembic 0005 created them.
    __table_args__ = (
        Index("ix_saved_notes_conversation", "conversation_id"),
        Index("ix_saved_notes_collection", "collection_id"),
    )
