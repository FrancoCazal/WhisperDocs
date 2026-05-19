from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    user_number: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    channel: Mapped[str] = mapped_column(
        String(20), default="web", server_default="web", nullable=False, index=True
    )
    collection_id: Mapped[int | None] = mapped_column(
        ForeignKey("collections.id", ondelete="SET NULL"), nullable=True
    )
    is_saved: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False, index=True
    )
    # Intentionally NO onupdate=func.now(). last_active_at is set explicitly
    # from handlers that represent real user activity (POST /agent/chat,
    # WhatsApp webhook with message). Admin row edits (e.g. PATCH is_saved)
    # do not refresh it. See phase-4-design.md §2.6.
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
