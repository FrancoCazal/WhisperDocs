"""Token streaming in the generation nodes (app/agent/utils/streaming.py,
audit A19): the events the SSE endpoint forwards, and what `stream_generate`
returns when streaming is off, outside a graph run, or cut short."""

from __future__ import annotations

import dataclasses

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_agent_graph
from app.agent.utils import streaming as streaming_module
from app.agent.utils.streaming import TOKEN_EVENT_NAME, stream_generate
from app.config import settings
from app.services.llm.base import LLMResponse, LLMStreamDelta

FINAL = LLMResponse(text="Hello", tokens_input=12, tokens_output=3, model="m", latency_ms=40)
REQUEST = {
    "system": "s",
    "messages": [{"role": "user", "content": "hi"}],
    "model": "m",
    "temperature": 0.0,
    "max_tokens": 50,
}


class _Streamer:
    """Streams `deltas`, then a last delta with `final` unless it is None.
    `fail_after` raises once that many deltas went out."""

    def __init__(self, deltas, final=FINAL, fail_after: int | None = None):
        self.deltas = deltas
        self.final = final
        self.fail_after = fail_after
        self.calls: list[str] = []

    async def generate(self, **kwargs):
        self.calls.append("generate")
        assert kwargs == REQUEST
        return self.final

    async def generate_stream(self, **kwargs):
        self.calls.append("generate_stream")
        assert kwargs == REQUEST
        for sent, delta in enumerate(self.deltas):
            if sent == self.fail_after:
                raise TimeoutError("stream cut")
            yield LLMStreamDelta(text_delta=delta)
        if self.final is not None:
            yield LLMStreamDelta(text_delta="", final=self.final)


@pytest.fixture(autouse=True)
def _streaming_on(monkeypatch):
    monkeypatch.setattr(settings.agent, "streaming_tokens", True)


async def _in_a_run(provider, node: str = "answer_direct"):
    """`stream_generate` inside a runnable, as in a graph node: its response
    and the token events the run emitted."""
    got: list[LLMResponse] = []

    async def generate(_input):
        got.append(await stream_generate(provider, **REQUEST, node=node))
        return "done"

    events = [e async for e in RunnableLambda(generate).astream_events({}, version="v2")]
    tokens = [
        e["data"]
        for e in events
        if e["event"] == "on_custom_event" and e["name"] == TOKEN_EVENT_NAME
    ]
    return got[0], tokens


async def test_each_token_is_an_event_naming_its_node():
    provider = _Streamer(["Hel", "", "lo"])

    response, tokens = await _in_a_run(provider, node="answer_direct")

    # Empty deltas send no event.
    assert tokens == [
        {"node": "answer_direct", "delta": "Hel"},
        {"node": "answer_direct", "delta": "lo"},
    ]
    # The final delta's response, with its usage, not one rebuilt from the text.
    assert response == FINAL
    assert provider.calls == ["generate_stream"]


async def test_with_streaming_off_it_makes_one_plain_call(monkeypatch):
    monkeypatch.setattr(settings.agent, "streaming_tokens", False)
    provider = _Streamer(["Hel", "lo"])

    response, tokens = await _in_a_run(provider)

    assert response == FINAL
    assert tokens == []
    assert provider.calls == ["generate"]


async def test_outside_a_graph_run_it_answers_and_stops_trying_to_dispatch(monkeypatch):
    attempts: list[str] = []

    async def no_parent_run(name, data):  # noqa: ARG001
        attempts.append(data["delta"])
        raise RuntimeError("Unable to dispatch an adhoc event without a parent run id.")

    monkeypatch.setattr(streaming_module, "adispatch_custom_event", no_parent_run)

    response = await stream_generate(_Streamer(["a", "b", "c"]), **REQUEST, node="search_docs")

    assert response == FINAL
    assert attempts == ["a"]


async def test_a_stream_without_a_final_delta_returns_the_text_it_got():
    response, _ = await _in_a_run(_Streamer(["Hel", "lo"], final=None))

    assert response == LLMResponse(
        text="Hello", tokens_input=0, tokens_output=0, model="m", latency_ms=0
    )


async def test_a_stream_cut_midway_raises_after_the_tokens_it_sent():
    """The endpoint has shown those tokens; it sends `reset` on the error."""
    provider = _Streamer(["Half", " an", " answer"], fail_after=1)
    tokens: list[str] = []

    async def generate(_input):
        return await stream_generate(provider, **REQUEST, node="search_docs")

    with pytest.raises(TimeoutError):
        async for e in RunnableLambda(generate).astream_events({}, version="v2"):
            if e["event"] == "on_custom_event" and e["name"] == TOKEN_EVENT_NAME:
                tokens.append(e["data"]["delta"])

    assert tokens == ["Half"]


async def test_the_graph_streams_a_direct_answer_token_by_token(
    fake_provider, classifier_response, monkeypatch
):
    """The events `/agent/chat/stream` turns into `token` frames come out of
    the compiled graph with the node that wrote them."""
    fake_provider.structured_response = classifier_response("answer_direct")

    async def two_tokens(**_):
        yield LLMStreamDelta(text_delta="¡Hola")
        yield LLMStreamDelta(text_delta=" Franco!")
        yield LLMStreamDelta(text_delta="", final=dataclasses.replace(FINAL, text="¡Hola Franco!"))

    fake_provider.generate_stream = two_tokens  # type: ignore[assignment]
    for module in ("app.agent.nodes.decide_action", "app.agent.nodes.answer_direct"):
        monkeypatch.setattr(f"{module}.llm_service_for", lambda _n: (fake_provider, "m", 0.0))
    agent = build_agent_graph().compile(checkpointer=MemorySaver())

    events = [
        e
        async for e in agent.astream_events(
            {"messages": [HumanMessage(content="hola")], "collection_id": None},
            config={"configurable": {"thread_id": "t-stream"}, "recursion_limit": 10},
            version="v2",
        )
    ]

    tokens = [
        e["data"]
        for e in events
        if e["event"] == "on_custom_event" and e["name"] == TOKEN_EVENT_NAME
    ]
    assert tokens == [
        {"node": "answer_direct", "delta": "¡Hola"},
        {"node": "answer_direct", "delta": " Franco!"},
    ]
    state = await agent.aget_state({"configurable": {"thread_id": "t-stream"}})
    assert state.values["messages"][-1].content == "¡Hola Franco!"
