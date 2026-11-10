"""The reads that rank chunks or list documents leave each document's text in
the database (audit R10): it can run to megabytes, and none of them shows it.

Needs Postgres, like test_retrieval_db.py: point TEST_DATABASE_URL at a
database migrated to head. Each test runs inside a transaction that is rolled
back.
"""

from __future__ import annotations

import os

import httpx
import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.config import settings
from app.db import get_db
from app.main import app
from app.models import Chunk, Collection, Document
from app.services.catalog import list_documents
from app.services.retriever import VectorRetriever

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")

_EMBEDDING = [0.0] * (settings.embedding_dimension - 1) + [1.0]
_TEXTS = ["HNSW indexes trade memory for recall.", "IVFFlat clusters the vectors first."]


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


async def _document(session: AsyncSession) -> Document:
    coll = Collection(name="document-text")
    session.add(coll)
    await session.flush()
    doc = Document(
        collection_id=coll.id,
        title="indexes.md",
        filename="indexes.md",
        file_type="md",
        raw_text="\n".join(_TEXTS),
        language="en",
    )
    session.add(doc)
    await session.flush()
    session.add_all(
        Chunk(
            document_id=doc.id,
            chunk_index=i,
            text=t,
            token_count=len(t.split()),
            embedding=_EMBEDDING,
            language="en",
        )
        for i, t in enumerate(_TEXTS)
    )
    await session.flush()
    return doc


def _reads_text(statements: list[str]) -> list[str]:
    return [s for s in statements if "raw_text" in s]


async def test_ranking_chunks_reads_no_document_text(db):
    session, statements = db
    doc = await _document(session)
    statements.clear()

    # Fewer chunks than top_k in a collection: the vector leg runs twice, on
    # the index and then exactly (VectorRetriever.retrieve).
    chunks = await VectorRetriever(session).hybrid_retrieve(
        "HNSW recall", _EMBEDDING, collection_id=doc.collection_id, top_k=5
    )

    assert sorted((c.document_id, c.document_title, c.chunk_index, c.text) for c in chunks) == [
        (doc.id, "indexes.md", i, t) for i, t in enumerate(_TEXTS)
    ]
    assert all(c.cosine == 1.0 for c in chunks)
    assert _reads_text(statements) == []


async def test_listing_documents_reads_no_document_text(db):
    session, statements = db
    doc = await _document(session)
    statements.clear()

    page = await list_documents(session, collection_id=doc.collection_id, limit=10, cursor=None)

    assert [(d.id, d.title) for d in page.items] == [(doc.id, "indexes.md")]
    assert _reads_text(statements) == []


async def test_listing_a_documents_chunks_reads_no_document_text(db):
    session, statements = db
    doc = await _document(session)
    statements.clear()

    async def _db():
        yield session

    app.dependency_overrides[get_db] = _db
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as client:
            listed = await client.get(f"/documents/{doc.id}/chunks")
            missing = await client.get(f"/documents/{doc.id + 1000}/chunks")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert listed.status_code == 200
    assert [c["text"] for c in listed.json()] == _TEXTS
    assert missing.status_code == 404
    assert _reads_text(statements) == []
