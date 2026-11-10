"""The agent's turns at their edges (audit A8): every node of the graph in the
stream, a `reset` when the answer is not what was streamed, a stream the
client closes, and one turn at a time per thread.

Runs the real app over ASGI, without its lifespan, with the database and the
compiled graph replaced.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import structlog
from langchain_core.messages import AIMessage, HumanMessage
from opentelemetry.trace import StatusCode
from starlette.requests import Request

from app.agent.graph import NODE_NAMES, build_agent_graph
from app.agent.thread_lock import threads_with_turns
from app.agent.utils.streaming import TOKEN_EVENT_NAME
from app.api.v1.agent import agent_chat_stream
from app.db import get_db
from app.main import app
from app.models import Conversation
from app.schemas.agent import AgentChatRequest

THREAD = "5b0a4c7e-9a43-4a59-9a7e-0d5f3f1b2c11"


class _FakeDb:
    async def scalar(self, _statement):
        return Conversation(thread_id=THREAD, collection_id=None, channel="web")

    def add(self, _obj) -> None:
        pass

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        pass


@pytest.fixture(autouse=True)
def db(monkeypatch):
    fake = _FakeDb()

    async def fake_db():
        yield fake

    class _Session:
        async def __aenter__(self):
            return fake

        async def __aexit__(self, *_):
            return False

    monkeypatch.setattr("app.api.v1.agent.SessionLocal", _Session)
    app.dependency_overrides[get_db] = fake_db
    yield fake
    app.dependency_overrides.clear()


def _token(delta: str, node: str = "search_docs") -> dict:
    return {
        "event": "on_custom_event",
        "name": TOKEN_EVENT_NAME,
        "data": {"node": node, "delta": delta},
    }


def _state(answer: str) -> dict:
    return {
        "messages": [HumanMessage(content="q"), AIMessage(content=answer)],
        "pending_action": "search_docs",
        "retrieved_chunks": [],
        "metadata": {},
    }


class _ScriptedAgent:
    """Streams `events`, then raises `error` or ends with `answer`."""

    def __init__(
        self, events: list[dict], answer: str = "a", error: Exception | None = None
    ) -> None:
        self.events = events
        self.answer = answer
        self.error = error

    async def astream_events(self, _input, config, version):
        for event in self.events:
            yield event
        if self.error is not None:
            raise self.error
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": _state(self.answer)},
        }


async def _stream(agent) -> list[tuple[str, dict]]:
    app.state.agent = agent
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
            r = await c.post("/agent/chat/stream", json={"message": "q"})
    finally:
        del app.state.agent
    out = []
    for block in r.text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def _names(events: list[tuple[str, dict]]) -> list[str]:
    return [name for name, _ in events]


async def test_every_node_of_the_graph_is_reported():
    graph_nodes = set(build_agent_graph().nodes) - {"__start__"}
    script = []
    for node in sorted(graph_nodes):
        script.append({"event": "on_chain_start", "name": node})
        script.append({"event": "on_chain_end", "name": node, "data": {"output": {}}})

    events = await _stream(_ScriptedAgent(script))

    assert NODE_NAMES == graph_nodes
    assert "out_of_scope" in graph_nodes
    assert [d["node"] for n, d in events if n == "node_start"] == sorted(graph_nodes)
    assert [d["node"] for n, d in events if n == "node_end"] == sorted(graph_nodes)


async def test_an_answer_that_is_what_was_streamed_sends_no_reset():
    events = await _stream(_ScriptedAgent([_token("Hel"), _token("lo\n")], answer="Hello"))

    assert _names(events) == ["thread", "token", "token", "final"]


async def test_an_answer_other_than_the_streamed_one_comes_after_a_reset():
    """A generation that fails partway: the node answers "I had a problem"."""
    events = await _stream(_ScriptedAgent([_token("Half an ans")], answer="I had a problem."))

    assert _names(events) == ["thread", "token", "reset", "final"]
    assert events[2][1] == {"node": "search_docs"}
    assert events[3][1]["answer"] == "I had a problem."


async def test_an_error_after_tokens_comes_after_a_reset():
    events = await _stream(_ScriptedAgent([_token("Half")], error=RuntimeError("boom")))

    assert _names(events) == ["thread", "token", "reset", "error"]


async def test_an_error_before_any_token_sends_no_reset():
    events = await _stream(_ScriptedAgent([], error=RuntimeError("boom")))

    assert _names(events) == ["thread", "error"]


class _Hangs:
    """Streams one token, then waits for an answer that never comes."""

    async def astream_events(self, _input, config, version):
        yield {"event": "on_chain_start", "name": "search_docs"}
        yield _token("Half an ans")
        await asyncio.Event().wait()


async def test_a_client_that_disconnects_cancels_the_turn_and_it_is_logged(spans):
    """Over ASGI, as the server sees a closed connection: Starlette cancels
    the stream."""
    app.state.agent = _Hangs()
    token_sent = asyncio.Event()
    body = json.dumps({"message": "q", "thread_id": THREAD}).encode()
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        if messages:
            return messages.pop()
        await token_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and b"event: token" in message.get("body", b""):
            token_sent.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/agent/chat/stream",
        "raw_path": b"/api/v1/agent/chat/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"test"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    try:
        with structlog.testing.capture_logs() as logs:
            await asyncio.wait_for(app(scope, receive, send), timeout=5)
    finally:
        del app.state.agent

    (cut,) = [e for e in logs if e["event"] == "agent_stream_cancelled"]
    assert cut["thread_id"] == THREAD
    assert cut["node"] == "search_docs"
    assert cut["streamed_chars"] == len("Half an ans")
    (turn,) = [s for s in spans.get_finished_spans() if s.name.startswith("invoke_workflow")]
    assert turn.attributes["error.type"] == "cancelled"
    assert turn.status.status_code == StatusCode.ERROR
    assert threads_with_turns() == set()


async def test_a_stream_closed_between_events_is_logged_too():
    """The other way a stream ends early: closed while it waits at a yield."""
    app.state.agent = _Hangs()
    request = Request({"type": "http", "app": app, "headers": [], "state": {}})
    try:
        response = await agent_chat_stream(AgentChatRequest(message="q"), request)
        stream = response.body_iterator
        with structlog.testing.capture_logs() as logs:
            assert b"event: thread" in await anext(stream)
            assert b"event: node_start" in await anext(stream)
            await stream.aclose()
    finally:
        del app.state.agent

    (cut,) = [e for e in logs if e["event"] == "agent_stream_cancelled"]
    assert cut["node"] == "search_docs"
    assert cut["streamed_chars"] == 0
    assert threads_with_turns() == set()


class _Records:
    """Answers after a pause, noting when each turn starts and ends."""

    def __init__(self) -> None:
        self.log: list[str] = []

    async def _turn(self, _input) -> dict:
        question = _input["messages"][0].content
        self.log.append(f"start {question}")
        await asyncio.sleep(0.02)
        self.log.append(f"end {question}")
        return _state("a")

    async def ainvoke(self, _input, config):
        return await self._turn(_input)

    async def astream_events(self, _input, config, version):
        state = await self._turn(_input)
        yield {"event": "on_chain_end", "name": "LangGraph", "data": {"output": state}}


@pytest.mark.parametrize("path", ["/agent/chat", "/agent/chat/stream"])
async def test_two_turns_on_one_thread_run_one_after_the_other(path):
    agent = _Records()
    app.state.agent = agent
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
            first, second, other = await asyncio.gather(
                c.post(path, json={"message": "one", "thread_id": THREAD}),
                c.post(path, json={"message": "two", "thread_id": THREAD}),
                c.post(path, json={"message": "elsewhere"}),
            )
    finally:
        del app.state.agent

    assert first.status_code == second.status_code == other.status_code == 200
    mine = [e for e in agent.log if not e.endswith("elsewhere")]
    assert mine == ["start one", "end one", "start two", "end two"]
    # A turn on another thread does not wait.
    assert agent.log.index("start elsewhere") < agent.log.index("end one")
    assert threads_with_turns() == set()
