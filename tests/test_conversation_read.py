"""`GET /conversations/{thread_id}`: a thread opened from a link (frontend
review U3) gets its channel and collection without paging the list.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from app.db import get_db
from app.main import app
from app.models import Conversation


class _FakeDb:
    def __init__(self, conv: Conversation | None) -> None:
        self.conv = conv

    async def scalar(self, _statement):
        return self.conv


@pytest.fixture
def use_db():
    def install(conv: Conversation | None) -> None:
        async def fake_db():
            yield _FakeDb(conv)

        app.dependency_overrides[get_db] = fake_db

    yield install
    app.dependency_overrides.clear()


async def _get(path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
        return await c.get(path)


async def test_a_thread_is_read_by_its_id(use_db):
    now = datetime(2026, 11, 7, tzinfo=UTC)
    use_db(
        Conversation(
            id=7,
            thread_id="wa-thread",
            user_number="+5491100000000",
            channel="whatsapp",
            collection_id=3,
            is_saved=False,
            last_active_at=now,
            created_at=now,
        )
    )

    r = await _get("/conversations/wa-thread")

    assert r.status_code == 200
    assert r.json()["channel"] == "whatsapp"
    assert r.json()["collection_id"] == 3


async def test_an_unknown_thread_is_a_404(use_db):
    use_db(None)

    r = await _get("/conversations/nope")

    assert r.status_code == 404
