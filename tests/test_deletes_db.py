"""Deleting and moving documents against a real Postgres (audit R11).

The foreign keys' ON DELETE CASCADE removes a collection's documents and a
document's chunks, so none of these loads them first: a document's chunks
carry their embeddings, and a collection can hold thousands.

Needs Postgres, like test_retrieval_db.py: point TEST_DATABASE_URL at a
database migrated to head. Each test runs inside a transaction that is rolled
back.
"""

from __future__ import annotations

import os

import httpx
import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.config import settings
from app.db import get_db
from app.main import app
from app.models import Chunk, Collection, Document

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")

_EMBEDDING = [0.0] * (settings.embedding_dimension - 1) + [1.0]


@pytest.fixture
async def api():
    """A client, its session, and the SQL sent from the moment a test clears it."""
    engine = create_async_engine(DB_URL)
    statements: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    async with engine.connect() as conn:
        outer = await conn.begin()
        async with AsyncSession(bind=conn, expire_on_commit=False) as session:

            async def _db():
                yield session

            app.dependency_overrides[get_db] = _db
            transport = httpx.ASGITransport(app=app)
            try:
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test/api/v1"
                ) as client:
                    yield client, session, statements
            finally:
                app.dependency_overrides.pop(get_db, None)
        await outer.rollback()
    await engine.dispose()


async def _collection(session: AsyncSession, name: str, documents: int) -> tuple[int, list[int]]:
    """A collection of `documents` documents, two chunks each."""
    coll = Collection(name=name)
    session.add(coll)
    await session.flush()
    ids = []
    for i in range(documents):
        doc = Document(
            collection_id=coll.id,
            title=f"{name}-{i}",
            filename=f"{name}-{i}.md",
            file_type="md",
            raw_text="one\ntwo",
        )
        session.add(doc)
        await session.flush()
        session.add_all(
            Chunk(document_id=doc.id, chunk_index=j, text=t, token_count=1, embedding=_EMBEDDING)
            for j, t in enumerate(["one", "two"])
        )
        ids.append(doc.id)
    await session.flush()
    # As a request finds it: nothing loaded yet.
    session.expunge_all()
    return coll.id, ids


async def _chunks(session: AsyncSession, document_ids: list[int]) -> int:
    return await session.scalar(
        select(func.count(Chunk.id)).where(Chunk.document_id.in_(document_ids))
    )


def _reads(statements: list[str], table: str) -> list[str]:
    return [s for s in statements if s.lstrip().startswith("SELECT") and f"FROM {table}" in s]


async def test_deleting_a_collection_loads_none_of_its_documents(api):
    client, session, statements = api
    cid, doc_ids = await _collection(session, "delete-collection", 2)
    statements.clear()

    r = await client.delete(f"/collections/{cid}")

    assert r.status_code == 204
    assert _reads(statements, "documents") == []
    assert _reads(statements, "chunks") == []
    assert await session.get(Collection, cid) is None
    assert (
        await session.scalar(select(func.count(Document.id)).where(Document.id.in_(doc_ids))) == 0
    )
    assert await _chunks(session, doc_ids) == 0


async def test_the_orm_leaves_a_documents_chunks_to_the_foreign_key(api):
    """The models' side: a document loaded and deleted through the session,
    as a collection's loaded documents are, does not load its chunks."""
    _, session, statements = api
    _, (doc_id,) = await _collection(session, "orm-delete", 1)
    document = await session.get(Document, doc_id)
    statements.clear()

    await session.delete(document)
    await session.flush()

    assert _reads(statements, "chunks") == []
    assert await _chunks(session, [doc_id]) == 0


async def test_deleting_a_document_loads_neither_it_nor_its_chunks(api):
    client, session, statements = api
    _, (doc_id, other_id) = await _collection(session, "delete-document", 2)
    statements.clear()

    r = await client.delete(f"/documents/{doc_id}")
    again = await client.delete(f"/documents/{doc_id}")

    assert (r.status_code, again.status_code) == (204, 404)
    assert _reads(statements, "documents") == []
    assert _reads(statements, "chunks") == []
    assert await _chunks(session, [doc_id]) == 0
    assert await _chunks(session, [other_id]) == 2


async def test_bulk_delete_is_one_statement(api):
    client, session, statements = api
    _, (a, b, kept) = await _collection(session, "bulk-delete", 3)
    missing = kept + 1000
    statements.clear()

    r = await client.post("/documents/bulk-delete", json={"document_ids": [a, missing, b]})

    assert r.json() == {
        "requested": 3,
        "succeeded": 2,
        "failed": 1,
        "not_found": [missing],
        "errors": {},
    }
    assert [s.split()[0] for s in statements if "documents" in s] == ["DELETE"]
    assert _reads(statements, "chunks") == []
    assert await _chunks(session, [a, b]) == 0
    assert await _chunks(session, [kept]) == 2


async def test_bulk_move_is_one_statement(api):
    client, session, statements = api
    _, (a, b, kept) = await _collection(session, "bulk-move-from", 3)
    target, _ = await _collection(session, "bulk-move-to", 0)
    missing = kept + 1000
    statements.clear()

    r = await client.post(
        "/documents/bulk-move",
        json={"document_ids": [a, b, missing], "target_collection_id": target},
    )
    no_target = await client.post(
        "/documents/bulk-move", json={"document_ids": [a], "target_collection_id": target + 1000}
    )

    assert r.json() == {
        "requested": 3,
        "succeeded": 2,
        "failed": 1,
        "not_found": [missing],
        "errors": {},
    }
    assert [s.split()[0] for s in statements if "documents" in s] == ["UPDATE"]
    assert no_target.status_code == 404
    moved = await session.scalars(select(Document.id).where(Document.collection_id == target))
    assert sorted(moved.all()) == sorted([a, b])
    assert await _chunks(session, [a, b]) == 4
