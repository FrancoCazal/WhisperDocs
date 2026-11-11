"""Pruning a thread's checkpoints against Postgres (audit N6): the real graph
and saver, a fake model. The thread keeps its state and takes its next turn,
and the other threads keep theirs.

Needs Postgres, like test_history_filters_db.py: point TEST_DATABASE_URL at a
database migrated to head. The test's threads are deleted at the end; it
prunes only them.
"""

from __future__ import annotations

import os
import uuid

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import tuple_row
from psycopg_pool import AsyncConnectionPool

from app.agent.checkpointer import Pruned, prune_threads, threads_to_prune
from app.agent.graph import build_agent_graph
from app.agent.schemas import AgentAction
from app.agent.utils.messages import asked
from app.config import settings
from app.services.llm.base import LLMResponse, LLMStreamDelta

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set"),
    # The saver drives psycopg's async mode (tests/conftest.py).
    pytest.mark.psycopg,
]

_REPLY = LLMResponse(
    text="la respuesta", tokens_input=10, tokens_output=5, model="fake", latency_ms=1
)


class _Model:
    async def generate_structured(self, **_):
        return AgentAction(action="answer_direct", reason="t", confidence="high"), _REPLY

    async def generate(self, **_):
        return _REPLY

    async def generate_stream(self, **_):
        yield LLMStreamDelta(text_delta=_REPLY.text, final=None)
        yield LLMStreamDelta(text_delta="", final=_REPLY)


@pytest.fixture
async def saver():
    pool = AsyncConnectionPool(
        conninfo=(DB_URL or "").replace("+asyncpg", ""),
        kwargs={"autocommit": True},
        open=False,
    )
    await pool.open()
    try:
        saver = AsyncPostgresSaver(conn=pool)  # type: ignore[arg-type]
        await saver.setup()
        yield saver
    finally:
        await pool.close()


@pytest.fixture
def agent(saver, monkeypatch):
    model = _Model()
    for module in ("app.agent.nodes.decide_action", "app.agent.nodes.answer_direct"):
        monkeypatch.setattr(f"{module}.llm_service_for", lambda _n: (model, "fake", 0.0))
    monkeypatch.setattr(settings.agent, "compaction_count_threshold", 1000)
    return build_agent_graph().compile(checkpointer=saver)


async def _turn(agent, thread_id: str, text: str) -> None:
    await agent.ainvoke(
        {"messages": [asked(text, collection_id=None)], "collection_id": None, "document_id": None},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
    )


async def _rows(saver, thread_id: str) -> tuple[int, int, int]:
    """Checkpoints, writes and `messages` blobs of a thread."""
    async with saver.conn.connection() as conn, conn.cursor(row_factory=tuple_row) as cur:
        await cur.execute(
            "SELECT (SELECT count(*) FROM checkpoints WHERE thread_id = %(t)s),"
            " (SELECT count(*) FROM checkpoint_writes WHERE thread_id = %(t)s),"
            " (SELECT count(*) FROM checkpoint_blobs WHERE thread_id = %(t)s"
            "  AND channel = 'messages')",
            {"t": thread_id},
        )
        return await cur.fetchone()


async def test_a_pruned_thread_keeps_its_state_and_takes_its_next_turn(agent, saver):
    pruned, other = (f"test-prune-{uuid.uuid4()}" for _ in range(2))
    try:
        for i in range(3):
            await _turn(agent, pruned, f"q{i}")
        await _turn(agent, other, "q")
        config = {"configurable": {"thread_id": pruned}}
        before = await agent.aget_state(config)
        checkpoints, _, messages_copies = await _rows(saver, pruned)
        other_before = await _rows(saver, other)
        assert checkpoints > 3 and messages_copies > 3
        assert {pruned, other} <= set(await threads_to_prune(saver))

        assert await prune_threads(saver, [pruned]) == Pruned(threads=1)

        assert await _rows(saver, pruned) == (1, 0, 1)
        after = await agent.aget_state(config)
        assert after.values == before.values
        assert len(after.values["messages"]) == 6
        assert await _rows(saver, other) == other_before
        candidates = set(await threads_to_prune(saver))
        assert pruned not in candidates and other in candidates

        await _turn(agent, pruned, "q3")

        state = await agent.aget_state(config)
        assert [m.content for m in state.values["messages"]][-2:] == ["q3", "la respuesta"]
        assert len(state.values["messages"]) == 8
        assert pruned in set(await threads_to_prune(saver))
    finally:
        for thread_id in (pruned, other):
            await saver.adelete_thread(thread_id)
