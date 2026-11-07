"""`/agent/chat` logs no query for a failed turn (audit N4).

Runs the real app over ASGI, without its lifespan, with the database and the
compiled graph replaced.
"""

from __future__ import annotations

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.db import get_db
from app.main import app
from app.models import Conversation, QueryLog


class _FakeDb:
    def __init__(self) -> None:
        self.added: list[object] = []

    async def scalar(self, _statement):
        return Conversation(thread_id="t", collection_id=None, channel="web")

    def add(self, obj) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        pass


class _FakeAgent:
    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata

    async def ainvoke(self, _input, config):
        return {
            "messages": [HumanMessage(content="q"), AIMessage(content="a")],
            "pending_action": "search_docs",
            "retrieved_chunks": [],
            "metadata": self.metadata,
        }


@pytest.fixture
def db():
    fake = _FakeDb()

    async def fake_db():
        yield fake

    app.dependency_overrides[get_db] = fake_db
    yield fake
    app.dependency_overrides.clear()


@pytest.mark.parametrize(("turn_failed", "logged"), [(False, 1), (True, 0)])
async def test_query_log_only_for_answers(db, turn_failed, logged):
    app.state.agent = _FakeAgent({"turn_failed": turn_failed, "retrieval_chunks_count": 1})
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
            r = await c.post("/agent/chat", json={"message": "q"})
    finally:
        del app.state.agent

    assert r.status_code == 200
    assert r.json()["answer"] == "a"
    assert sum(isinstance(o, QueryLog) for o in db.added) == logged
