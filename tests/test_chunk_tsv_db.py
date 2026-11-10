"""`chunks.tsv` in the model (audit R17): the column alembic 0013 generates,
declared so autogenerate and `alembic check` see it, and never read by the ORM.

Needs Postgres, like test_retrieval_db.py: point TEST_DATABASE_URL at a
database migrated to head. Each test runs inside a transaction that is rolled
back.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.config import settings
from app.models import Chunk, Collection, Document
from app.models.chunk import TSV_EXPRESSION

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture
async def db():
    """A session, and the SQL it sends from the moment the test clears it."""
    engine = create_async_engine(DB_URL)
    statements: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    async with engine.connect() as conn:
        outer = await conn.begin()
        async with AsyncSession(bind=conn, expire_on_commit=False) as session:
            yield session, statements
        await outer.rollback()
    await engine.dispose()


async def _document(session: AsyncSession) -> int:
    coll = Collection(name="chunk-tsv")
    session.add(coll)
    await session.flush()
    doc = Document(collection_id=coll.id, title="t", filename="t.md", file_type="md", raw_text="x")
    session.add(doc)
    await session.flush()
    return doc.id


async def test_the_model_says_what_postgres_generates(db):
    session, _ = db
    generated = await session.scalar(
        text(
            "SELECT pg_get_expr(d.adbin, d.adrelid) FROM pg_attrdef d"
            " JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum"
            " WHERE d.adrelid = 'chunks'::regclass AND a.attname = 'tsv'"
        )
    )

    assert " ".join(generated.split()) == " ".join(TSV_EXPRESSION.split())


async def test_storing_and_reading_chunks_leaves_tsv_in_the_database(db):
    session, statements = db
    doc_id = await _document(session)
    statements.clear()

    session.add_all(
        Chunk(
            document_id=doc_id,
            chunk_index=i,
            text="Postgres computes the lexemes",
            token_count=4,
            embedding=[0.0] * settings.embedding_dimension,
            language="en",
        )
        for i in range(2)
    )
    await session.flush()
    session.expunge_all()
    chunks = (await session.scalars(select(Chunk).where(Chunk.document_id == doc_id))).all()

    assert len(chunks) == 2
    assert [s for s in statements if "tsv" in s] == []
    # Postgres filled it all the same.
    lexemes = await session.scalar(
        text("SELECT tsv::text FROM chunks WHERE id = :id"), {"id": chunks[0].id}
    )
    assert "'lexem'" in lexemes
