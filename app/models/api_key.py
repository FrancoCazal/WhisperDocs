from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class ApiKey(Base):
    """A key for the MCP server over HTTP (audit M-2), scoped to collections.

    Only the key's SHA-256 is stored: the key itself is shown once, when
    `python -m app.mcp.keys create` makes it. Keys are random, so a plain hash
    is enough; there is no password to stretch.
    """

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # The key's first characters, to tell keys apart in a list.
    prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    # The collections the key reaches; null reaches all of them.
    collection_ids: Mapped[list[int] | None] = mapped_column(ARRAY(Integer), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
