from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

if TYPE_CHECKING:
    from app.models.chunk import Chunk
    from app.models.collection import Collection


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    collection_id: Mapped[int] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    filename: Mapped[str] = mapped_column(String(300), nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    # SHA-256 of `raw_text` (audit R12): a collection holds each document
    # once, which the unique index below keeps under concurrent uploads.
    # Null on the later copies of a document that alembic 0021 found twice
    # in one collection, and on rows written without it.
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    # How many of its chunks have `instruction_flags`.
    flagged_chunks: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    collection: Mapped["Collection"] = relationship(back_populates="documents")
    # passive_deletes: the foreign key's ON DELETE CASCADE removes the chunks.
    # Without it, deleting a document would first load every chunk, embedding
    # included, to delete each one (audit R11).
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        Index("uq_documents_collection_content", "collection_id", "content_hash", unique=True),
    )
