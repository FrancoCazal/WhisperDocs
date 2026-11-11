from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class AuditLog(Base):
    """One call that writes through the MCP server, and how it ended (BE-36,
    audit M-2): a document added, deleted or moved, or a write that was
    refused, declined by the user, or failed.

    The ids are plain integers, not foreign keys: the record outlives the
    document and the collection it names. A write that succeeds commits its
    row in the same transaction (`app/services/audit.py`).
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Who: an MCP key's name, "api-key" (the API's own key), "open" (no key,
    # with the API open) or "stdio" (the process a client started).
    actor: Mapped[str] = mapped_column(String(100), nullable=False)
    # mcp-http or mcp-stdio.
    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    # The tool: ingest_document, delete_document, move_document.
    action: Mapped[str] = mapped_column(String(30), nullable=False)
    # done, declined, cancelled, refused or failed.
    outcome: Mapped[str] = mapped_column(String(20), nullable=False)
    collection_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    document_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The rest: the document's title, the target collection, why it was
    # refused or failed.
    detail: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )

    __table_args__ = (Index("ix_audit_log_created_at", "created_at"),)
