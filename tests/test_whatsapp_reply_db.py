"""A WhatsApp message through the real handler and Postgres (audit A11), and
a database error that quotes none of its values.

Needs Postgres, like test_history_filters_db.py: point TEST_DATABASE_URL at a
database migrated to head. Each test runs inside a transaction that is rolled
back. The agent and Twilio are replaced.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import uuid
from contextlib import asynccontextmanager

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.agent.thread_id import whatsapp_thread_id
from app.config import settings
from app.models import Collection, Conversation
from app.services.whatsapp import inbound

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


@pytest.fixture
def sent(monkeypatch):
    messages: list[tuple[str, str]] = []

    async def send(to, body):
        messages.append((to, body))

    monkeypatch.setattr(inbound, "send_whatsapp_message", send)
    return messages


class _Agent:
    """Answers every turn, and keeps what each one got."""

    def __init__(self) -> None:
        self.turns: list[dict] = []

    async def ainvoke(self, state, config):
        self.turns.append(
            {
                "collection_id": state["collection_id"],
                "question": state["messages"][-1].content,
                "scope": state["messages"][-1].additional_kwargs.get("scope"),
                "thread_id": config["configurable"]["thread_id"],
            }
        )
        return {
            "messages": [AIMessage(content="Respuesta")],
            "pending_action": "answer_direct",
            "metadata": {},
            "retrieved_chunks": [],
        }


def _number() -> str:
    return "5959" + "".join(random.choices("0123456789", k=8))


async def test_a_new_number_gets_a_thread_and_an_answer(session, sent, caplog):
    number = _number()
    sender = f"whatsapp:+{number}"
    agent = _Agent()

    @asynccontextmanager
    async def sessions():
        yield session

    with caplog.at_level(logging.INFO):
        await inbound.spawn_reply(agent, sender=sender, body="¿Qué dice?", sessions=sessions)

    thread_id = whatsapp_thread_id(sender)
    conv = await session.scalar(select(Conversation).where(Conversation.thread_id == thread_id))
    assert conv is not None
    assert (conv.user_number, conv.channel, conv.title) == (number, "whatsapp", "¿Qué dice?")
    assert agent.turns == [
        {
            "collection_id": settings.default_collection_id,
            "question": "¿Qué dice?",
            "scope": {"collection_id": settings.default_collection_id, "document_id": None},
            "thread_id": thread_id,
        }
    ]
    ((to, body),) = sent
    assert to == sender
    assert body.startswith(inbound.NEW_THREAD_DISCLAIMER)
    assert body.endswith("Respuesta")
    assert number not in caplog.text


async def test_a_collection_switch_applies_to_the_question_sent_right_after(session, sent):
    sender = f"whatsapp:+{_number()}"
    name = f"contratos-{uuid.uuid4().hex[:8]}"
    collection = Collection(name=name)
    session.add(collection)
    await session.flush()
    agent = _Agent()

    @asynccontextmanager
    async def sessions():
        yield session

    await asyncio.gather(
        inbound.spawn_reply(agent, sender=sender, body=f"/coleccion {name}", sessions=sessions),
        inbound.spawn_reply(agent, sender=sender, body="¿Qué dice?", sessions=sessions),
    )

    assert [turn["collection_id"] for turn in agent.turns] == [collection.id]
    assert agent.turns[0]["scope"] == {"collection_id": collection.id, "document_id": None}
    assert sent[0] == (sender, f"Ahora consultando: {name}. ¿Qué querés saber?")
    assert sent[1][1] == "Respuesta"


async def test_a_failed_turn_gets_an_apology(session, sent):
    sender = f"whatsapp:+{_number()}"

    class _Failing:
        async def ainvoke(self, state, config):
            raise RuntimeError("boom")

    @asynccontextmanager
    async def sessions():
        yield session

    await inbound.spawn_reply(_Failing(), sender=sender, body="¿Qué dice?", sessions=sessions)

    assert sent == [(sender, inbound.APOLOGY)]


async def test_a_database_error_quotes_none_of_its_values():
    """The app's engine hides a statement's parameters: a log line with the
    traceback holds no number, question or document text."""
    from app.db import engine

    number = _number()
    try:
        async with engine.connect() as conn:
            with pytest.raises(DBAPIError) as raised:
                await conn.execute(
                    text("SELECT 1 / 0 WHERE CAST(:number AS text) <> ''"), {"number": number}
                )
    finally:
        await engine.dispose()

    assert "division by zero" in str(raised.value)
    assert number not in str(raised.value)
