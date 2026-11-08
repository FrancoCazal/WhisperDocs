"""`/agent/chat` logs no query for a failed turn (audit N4), and the API
prices each answer with the backend's table (frontend review FR-2).

Runs the real app over ASGI, without its lifespan, with the database and the
compiled graph replaced.
"""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.utils.messages import failed_answer
from app.db import get_db
from app.main import app
from app.models import Conversation, QueryLog
from app.services.llm.pricing import estimate_cost_usd

SONNET = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


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
    def __init__(self, metadata: dict, messages: list | None = None) -> None:
        self.metadata = metadata
        self.messages = messages or []

    async def ainvoke(self, _input, config):
        return {
            "messages": [HumanMessage(content="q"), AIMessage(content="a")],
            "pending_action": "search_docs",
            "retrieved_chunks": [],
            "metadata": self.metadata,
        }

    async def aget_state(self, _config):
        class _Snapshot:
            values = {"messages": self.messages}

        return _Snapshot()


@pytest.fixture
def db():
    fake = _FakeDb()

    async def fake_db():
        yield fake

    app.dependency_overrides[get_db] = fake_db
    yield fake
    app.dependency_overrides.clear()


@pytest.mark.parametrize(("turn_failed", "logged"), [(False, 1), (True, 0)])
async def test_query_log_only_for_answers_and_the_failure_is_flagged(db, turn_failed, logged):
    app.state.agent = _FakeAgent({"turn_failed": turn_failed, "retrieval_chunks_count": 1})
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
            r = await c.post("/agent/chat", json={"message": "q"})
    finally:
        del app.state.agent

    assert r.status_code == 200
    assert r.json()["answer"] == "a"
    assert r.json()["failed"] is turn_failed
    assert sum(isinstance(o, QueryLog) for o in db.added) == logged


async def _call(method: str, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
        return await c.request(method, path, **kwargs)


async def test_the_answer_cost_comes_priced_from_the_backend(db):
    app.state.agent = _FakeAgent(
        {
            "generation_model": SONNET,
            "generation_tokens_input": 3446,
            "generation_tokens_output": 406,
        }
    )
    try:
        r = await _call("POST", "/agent/chat", json={"message": "q"})
    finally:
        del app.state.agent

    expected = estimate_cost_usd(SONNET, 3446, 406)
    assert expected == 0.016428
    assert r.json()["metadata"]["cost_usd"] == expected
    (log,) = [o for o in db.added if isinstance(o, QueryLog)]
    assert log.cost_usd == expected


async def test_a_thread_prices_answers_written_before_the_cost_field(db):
    old_metadata = {"model": SONNET, "tokens_input": 3446, "tokens_output": 406}
    app.state.agent = _FakeAgent(
        {},
        messages=[
            HumanMessage(content="q"),
            AIMessage(content="a", additional_kwargs={"metadata": old_metadata}),
        ],
    )
    try:
        r = await _call("GET", "/conversations/t/messages")
    finally:
        del app.state.agent

    user, answer = r.json()
    assert user["metadata"] is None
    assert answer["metadata"]["cost_usd"] == estimate_cost_usd(SONNET, 3446, 406)


async def test_a_thread_flags_its_failed_answers(db):
    app.state.agent = _FakeAgent(
        {},
        messages=[
            HumanMessage(content="q"),
            failed_answer("I had a problem…", metadata={"model": SONNET}),
            HumanMessage(content="q again"),
            AIMessage(content="a"),
        ],
    )
    try:
        r = await _call("GET", "/conversations/t/messages")
    finally:
        del app.state.agent

    assert [m["failed"] for m in r.json()] == [False, True, False, False]


CLASSIFIER_BILLING = {"step": "classifier", "kind": "billing", "model": "gemini-3.8-flash"}


async def test_an_answer_says_which_calls_failed(db):
    """With Vertex refusing every call, the classifier falls back and the
    turn still answers: `errors` is what shows the outage (BE-11)."""
    app.state.agent = _FakeAgent({"turn_failed": False, "turn_errors": [CLASSIFIER_BILLING]})
    try:
        r = await _call("POST", "/agent/chat", json={"message": "q"})
    finally:
        del app.state.agent

    assert r.json()["failed"] is False
    assert r.json()["errors"] == [CLASSIFIER_BILLING]


class _StreamingAgent(_FakeAgent):
    async def astream_events(self, _input, config, version):
        state = await self.ainvoke(_input, config)
        yield {"event": "on_chain_end", "name": "LangGraph", "data": {"output": state}}


async def test_the_streams_final_event_says_which_calls_failed(db, monkeypatch):
    class _Session:
        async def __aenter__(self):
            return db

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr("app.api.v1.agent.SessionLocal", _Session)
    app.state.agent = _StreamingAgent({"turn_errors": [CLASSIFIER_BILLING]})
    try:
        r = await _call("POST", "/agent/chat/stream", json={"message": "q"})
    finally:
        del app.state.agent

    events = [block.split("\n", 1) for block in r.text.strip().split("\n\n")]
    (final,) = [data for name, data in events if name == "event: final"]
    payload = json.loads(final.removeprefix("data: "))
    assert payload["answer"] == "a"
    assert payload["errors"] == [CLASSIFIER_BILLING]


async def test_a_thread_says_which_calls_each_answer_had_fail(db):
    app.state.agent = _FakeAgent(
        {},
        messages=[
            HumanMessage(content="q"),
            AIMessage(content="a", additional_kwargs={"errors": [CLASSIFIER_BILLING]}),
            HumanMessage(content="q2"),
            # Written before the field.
            AIMessage(content="a2"),
        ],
    )
    try:
        r = await _call("GET", "/conversations/t/messages")
    finally:
        del app.state.agent

    assert [m["errors"] for m in r.json()] == [[], [CLASSIFIER_BILLING], [], []]
