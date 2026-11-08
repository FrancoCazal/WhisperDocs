"""The query log's filters: which answers it holds (the chat's or /query's)
and the text of their questions (frontend review U16).

Needs Postgres, like test_thread_lists_db.py: point TEST_DATABASE_URL at a
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
from app.models import QueryLog

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


def _log(question: str, action: str | None) -> QueryLog:
    return QueryLog(question=question, answer="a", model_used="m", agent_action=action)


async def test_the_log_filters_by_source_and_question(api):
    client, session = api
    tag = uuid.uuid4().hex[:8]
    session.add_all(
        [
            _log(f"{tag} chat at 50% recall", "search_docs"),
            _log(f"{tag} chat at 500 recall", "answer_direct"),
            _log(f"{tag} query on top_k", None),
        ]
    )
    await session.flush()

    async def questions(**params: str) -> tuple[list[str], int]:
        page = (await client.get("/query/history", params={"limit": 200, **params})).json()
        return [r["question"] for r in page["items"]], page["total"]

    chat, chat_total = await questions(q=tag, source="chat")
    query, query_total = await questions(q=tag, source="query")
    both, both_total = await questions(q=tag)

    assert sorted(chat) == sorted([f"{tag} chat at 500 recall", f"{tag} chat at 50% recall"])
    assert query == [f"{tag} query on top_k"]
    assert (chat_total, query_total, both_total) == (2, 1, 3)
    # "%" and "_" are the characters themselves, not LIKE wildcards.
    assert (await questions(q=f"{tag} chat at 50%"))[0] == [f"{tag} chat at 50% recall"]
    assert (await questions(q="TOP_K", source="query"))[0][:1] == [f"{tag} query on top_k"]


async def test_an_unknown_source_is_rejected(api):
    client, _ = api
    assert (await client.get("/query/history", params={"source": "whatsapp"})).status_code == 422
