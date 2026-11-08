"""The summary names the collections and documents its turns searched
(audit A17), looked up in Postgres.

Needs Postgres, like test_history_filters_db.py: point TEST_DATABASE_URL at a
database migrated to head. The test runs inside a transaction that is rolled
back.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.agent.nodes.summarize import _scope_names
from app.agent.utils.messages import asked
from app.models import Collection, Document

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")


@pytest.fixture
async def session():
    engine = create_async_engine(DB_URL)
    async with engine.connect() as conn:
        outer = await conn.begin()
        async with AsyncSession(bind=conn, expire_on_commit=False) as s:
            yield s
        await outer.rollback()
    await engine.dispose()


async def test_the_names_come_from_the_collections_and_documents(session):
    tag = uuid.uuid4().hex[:8]
    collection = Collection(name=f"Contratos {tag}")
    session.add(collection)
    await session.flush()
    document = Document(
        collection_id=collection.id,
        title=f"Manual {tag}",
        filename="manual.pdf",
        file_type="pdf",
        raw_text="x",
    )
    session.add(document)
    await session.flush()

    @asynccontextmanager
    async def sessions():
        yield session

    names = await _scope_names(
        [
            asked("a", collection_id=collection.id),
            asked("b", collection_id=collection.id, document_id=document.id),
            asked("c", collection_id=None),
            asked("d", collection_id=2_000_000_000),
        ],
        sessions=sessions,
    )

    assert names == {
        ("collection", collection.id): f"Contratos {tag}",
        ("document", document.id): f"Manual {tag}",
    }
