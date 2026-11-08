from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import ColumnElement
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, QueryableAttribute

from app.config import settings

# pool_pre_ping: after a Postgres restart, a pooled connection is dead; the
# ping replaces it on checkout instead of failing the request that drew it.
# hide_parameters: a database error's message leaves out the statement's
# values, which can be a WhatsApp number, a question or a document's text, so
# the traceback a log line carries holds none of them (audit A11).
engine = create_async_engine(
    settings.database_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
    hide_parameters=True,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


def contains_text(column: QueryableAttribute[Any], text: str) -> ColumnElement[bool]:
    """`column` contains `text`, ignoring case. The text is literal: the LIKE
    wildcards in it (% and _) match themselves."""
    literal = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return column.ilike(f"%{literal}%", escape="\\")


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
