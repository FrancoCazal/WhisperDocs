"""A thread remembers each turn's time and the node that answered it, and a
turn can move the thread to every collection (frontend review P3).

Runs the real app over ASGI, without its lifespan, with the database and the
compiled graph replaced.
"""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.utils.messages import stamped
from app.api.v1.agent import apply_turn_collection
from app.db import get_db
from app.main import app
from app.models import Conversation
from app.schemas.agent import AgentChatRequest

THREAD = "8f14e45f-ceea-467f-a0e6-0a8b7c1d2e3f"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"message": "q"}, 3),
        ({"message": "q", "collection_id": None}, None),
        ({"message": "q", "collection_id": 7}, 7),
    ],
    ids=["left-out-keeps-it", "null-is-all", "an-id-moves-it"],
)
def test_a_turn_moves_the_thread_only_when_it_names_a_collection(body, expected):
    conv = Conversation(thread_id="t", collection_id=3, channel="web")
    apply_turn_collection(conv, AgentChatRequest.model_validate(body))
    assert conv.collection_id == expected


class _Db:
    def __init__(self, conv: Conversation) -> None:
        self.conv = conv

    async def scalar(self, _statement):
        return self.conv

    def add(self, _obj) -> None:
        pass

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        pass


class _Agent:
    def __init__(self, messages: list | None = None) -> None:
        self.messages = messages or []
        self.inputs: list[dict] = []

    async def ainvoke(self, state, config):
        self.inputs.append(state)
        return {
            "messages": [*state["messages"], AIMessage(content="a")],
            "pending_action": "answer_direct",
            "retrieved_chunks": [],
            "metadata": {},
        }

    async def aget_state(self, _config):
        class _Snapshot:
            values = {"messages": self.messages}

        return _Snapshot()


@pytest.fixture
def conv():
    conv = Conversation(thread_id="t", collection_id=3, channel="web")

    async def fake_db():
        yield _Db(conv)

    app.dependency_overrides[get_db] = fake_db
    yield conv
    app.dependency_overrides.clear()


async def _call(agent: _Agent, method: str, path: str, **kwargs) -> httpx.Response:
    app.state.agent = agent
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
            return await c.request(method, path, **kwargs)
    finally:
        del app.state.agent


async def test_all_on_an_existing_thread_searches_every_collection(conv):
    agent = _Agent()
    r = await _call(
        agent,
        "POST",
        "/agent/chat",
        json={"message": "q", "thread_id": THREAD, "collection_id": None},
    )

    assert r.status_code == 200
    assert conv.collection_id is None
    assert agent.inputs[0]["collection_id"] is None


async def test_the_question_is_stamped_with_when_it_was_asked(conv):
    agent = _Agent()
    await _call(agent, "POST", "/agent/chat", json={"message": "q", "thread_id": THREAD})

    (question,) = agent.inputs[0]["messages"]
    assert datetime.fromisoformat(question.additional_kwargs["timestamp"]).tzinfo is not None


async def test_a_reopened_thread_has_each_turns_time_and_action(conv):
    agent = _Agent(
        [
            HumanMessage(content="q", additional_kwargs=stamped()),
            AIMessage(content="a", additional_kwargs=stamped(action="search_docs")),
            # Written before the stamp: no time and no action.
            HumanMessage(content="old q"),
            AIMessage(content="old a"),
        ]
    )
    r = await _call(agent, "GET", "/conversations/t/messages")

    question, answer, old_question, old_answer = r.json()
    assert question["timestamp"] is not None and question["action"] is None
    assert answer["timestamp"] is not None and answer["action"] == "search_docs"
    assert old_question["timestamp"] is None and old_answer["action"] is None
