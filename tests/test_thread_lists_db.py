"""The thread list's search, and notes that know their thread (frontend
review U11).

Needs Postgres, like test_list_totals_db.py: point TEST_DATABASE_URL at a
database migrated to head. Each test runs inside a transaction that is rolled
back.
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.db import get_db
from app.main import app
from app.models import Conversation, SavedNote

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


async def _thread(session: AsyncSession, title: str | None) -> Conversation:
    conv = Conversation(thread_id=str(uuid.uuid4()), channel="web", title=title)
    session.add(conv)
    await session.flush()
    return conv


async def test_the_search_matches_the_title_as_literal_text(api):
    client, session = api
    tag = uuid.uuid4().hex[:8]
    pct = await _thread(session, f"{tag} recall at 50% of top_k")
    await _thread(session, f"{tag} recall at 500 of topXk")
    await _thread(session, None)

    async def found(q: str) -> list[str]:
        page = (await client.get("/conversations", params={"q": q, "limit": 200})).json()
        return [c["thread_id"] for c in page["items"]]

    assert len(await found(tag)) == 2
    # "%" and "_" are the characters themselves, not LIKE wildcards.
    assert await found(f"{tag} recall at 50%") == [pct.thread_id]
    assert await found("TOP_K") == [pct.thread_id]
    page = (await client.get("/conversations", params={"q": tag})).json()
    assert page["total"] == 2
    assert page["items"][0]["title"].startswith(tag)


async def test_notes_carry_their_thread(api):
    client, session = api
    conv = await _thread(session, "a thread with a note")

    created = await client.post(f"/conversations/{conv.thread_id}/notes", json={"content": "x"})
    session.add(SavedNote(conversation_id=None, collection_id=None, source_message_content="y"))
    await session.flush()
    listed = (await client.get("/notes", params={"limit": 200})).json()
    of_thread = (await client.get("/notes", params={"conversation_id": conv.id})).json()

    assert created.status_code == 201
    assert created.json()["thread_id"] == conv.thread_id
    by_content = {n["source_message_content"]: n for n in listed}
    assert by_content["x"]["thread_id"] == conv.thread_id
    assert by_content["y"]["thread_id"] is None
    assert [n["thread_id"] for n in of_thread] == [conv.thread_id]
