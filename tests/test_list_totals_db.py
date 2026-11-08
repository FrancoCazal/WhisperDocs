"""Totals the server computes for lists (frontend review FR-17), so a client
holding one page never counts it as the whole.

Needs Postgres, like test_retrieval_db.py: point TEST_DATABASE_URL at a
database migrated to head. Each test runs inside a transaction that is rolled
back.
"""

from __future__ import annotations

import os

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.db import get_db
from app.main import app
from app.models import Collection, Document

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture
async def api():
    engine = create_async_engine(DB_URL)
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
                    yield client, session
            finally:
                app.dependency_overrides.pop(get_db, None)
        await outer.rollback()
    await engine.dispose()


async def _collection(session: AsyncSession, name: str, docs: list[tuple[int, int]]) -> int:
    """docs: (chunk_count, total_tokens) per document."""
    coll = Collection(name=name)
    session.add(coll)
    await session.flush()
    session.add_all(
        Document(
            collection_id=coll.id,
            title=f"{name}-{i}",
            filename=f"{name}-{i}.md",
            file_type="md",
            raw_text="x",
            chunk_count=chunks,
            total_tokens=tokens,
        )
        for i, (chunks, tokens) in enumerate(docs)
    )
    await session.flush()
    return coll.id


async def test_collections_carry_their_totals(api):
    client, session = api
    full = await _collection(session, "totals-full", [(3, 100), (2, 50), (4, 70)])
    empty = await _collection(session, "totals-empty", [])

    rows = {c["id"]: c for c in (await client.get("/collections")).json()}
    one = (await client.get(f"/collections/{full}")).json()

    totals = ("document_count", "chunk_count", "total_tokens")
    assert tuple(rows[full][k] for k in totals) == (3, 9, 220)
    assert tuple(rows[empty][k] for k in totals) == (0, 0, 0)
    assert tuple(one[k] for k in totals) == (3, 9, 220)


async def test_the_first_page_carries_the_total(api):
    client, session = api
    cid = await _collection(session, "totals-paged", [(1, 1)] * 5)

    first = (await client.get(f"/documents?collection_id={cid}&limit=2")).json()
    cursor = first["next_cursor"]
    second = (await client.get(f"/documents?collection_id={cid}&limit=2&cursor={cursor}")).json()

    assert (len(first["items"]), first["total"]) == (2, 5)
    # Later pages would only repeat it.
    assert (len(second["items"]), second["total"]) == (2, None)
