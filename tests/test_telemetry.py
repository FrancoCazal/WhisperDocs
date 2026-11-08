"""Traces (app/telemetry.py, audit M-3 and E11).

Spans go to an in-memory exporter (the `spans` fixture): nothing leaves the
process, and every model, embedding and database call is a fake.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
import structlog
from botocore.exceptions import ClientError
from langchain_core.messages import AIMessage, HumanMessage
from opentelemetry.trace import SpanKind, StatusCode
from pydantic import SecretStr

from app.agent.schemas import AgentAction
from app.config import settings
from app.db import get_db
from app.logging_config import add_trace_context
from app.main import app
from app.models import Conversation, QueryLog
from app.schemas.query import QueryRequest
from app.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.services.llm.base import LLMResponse, LLMStreamDelta
from app.services.llm.pricing import estimate_cost_usd
from app.services.llm.tracked import TrackedProvider
from app.services.retriever import RetrievedChunk
from app.telemetry import AGENT_WORKFLOW, build_tracer_provider, workflow

SONNET_4_5 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
SONNET_5_5 = "global.anthropic.claude-sonnet-5-5"
GEMINI = "gemini-3.8-flash"


class _Bedrock:
    """A provider double that names itself as BedrockProvider does."""

    otel_provider_name = "aws.bedrock"

    def __init__(
        self,
        response: LLMResponse | None = None,
        error: Exception | None = None,
        deltas: list[LLMStreamDelta] | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.deltas = deltas or []

    async def generate(self, **_kwargs) -> LLMResponse:
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response

    async def generate_structured(self, **_kwargs):
        assert self.response is not None
        return AgentAction(action="search_docs", reason="r", confidence="high"), self.response

    async def generate_stream(self, **_kwargs):
        for delta in self.deltas:
            yield delta


def _response(model: str, tokens_input: int, tokens_output: int) -> LLMResponse:
    return LLMResponse(
        text="an answer",
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        model=model,
        latency_ms=900,
    )


async def _call(provider: TrackedProvider, model: str, **kwargs) -> LLMResponse:
    return await provider.generate(
        system="s", messages=[{"role": "user", "content": "q"}], model=model, **kwargs
    )


# ---------- a model call ----------


async def test_a_model_call_is_a_client_span_with_its_request_and_usage(spans, monkeypatch):
    monkeypatch.setattr(settings, "bedrock_claude_effort", "low")
    provider = TrackedProvider(_Bedrock(_response(SONNET_5_5, 1200, 300)), "rag_generation")

    await _call(provider, SONNET_5_5, temperature=0.3, max_tokens=1024)

    (span,) = spans.get_finished_spans()
    assert span.name == f"chat {SONNET_5_5}"
    assert span.kind == SpanKind.CLIENT
    # Every attribute, so a prompt or an answer added by mistake fails here.
    # Claude 4.7+ is sent no temperature, so none is recorded.
    assert dict(span.attributes) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "aws.bedrock",
        "gen_ai.request.model": SONNET_5_5,
        "whisperdocs.role": "rag_generation",
        "gen_ai.request.max_tokens": 1024,
        "gen_ai.request.reasoning.level": "low",
        "gen_ai.request.stream": False,
        "gen_ai.response.model": SONNET_5_5,
        "gen_ai.usage.input_tokens": 1200,
        "gen_ai.usage.output_tokens": 300,
        "gen_ai.usage.cost": estimate_cost_usd(SONNET_5_5, 1200, 300),
    }
    assert span.status.status_code == StatusCode.UNSET


async def test_an_older_model_records_its_temperature_and_structured_output(spans):
    provider = TrackedProvider(_Bedrock(_response(SONNET_4_5, 10, 5)), "classifier")

    await provider.generate_structured(
        system="s",
        messages=[{"role": "user", "content": "q"}],
        model=SONNET_4_5,
        response_model=AgentAction,
        temperature=0.0,
        max_tokens=200,
    )

    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.request.temperature"] == 0.0
    assert span.attributes["gen_ai.output.type"] == "json"
    assert "gen_ai.request.reasoning.level" not in span.attributes


async def test_a_failed_call_records_its_kind_and_not_its_message(spans):
    throttled = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "quoting the prompt: secret"}},
        "Converse",
    )
    provider = TrackedProvider(_Bedrock(error=throttled), "rag_generation")

    with pytest.raises(ClientError):
        await _call(provider, SONNET_4_5)

    (span,) = spans.get_finished_spans()
    assert span.attributes["error.type"] == "rate_limit"
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "rate_limit"
    assert not span.events
    assert "secret" not in span.to_json()


async def test_a_stream_records_its_first_chunk_and_its_usage(spans):
    deltas = [
        LLMStreamDelta("Hola"),
        LLMStreamDelta(" mundo"),
        LLMStreamDelta("", final=_response(SONNET_4_5, 40, 2)),
    ]
    provider = TrackedProvider(_Bedrock(deltas=deltas), "answer_direct")

    async for _ in provider.generate_stream(
        system="s", messages=[{"role": "user", "content": "q"}], model=SONNET_4_5
    ):
        pass

    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.request.stream"] is True
    assert span.attributes["gen_ai.response.time_to_first_chunk"] >= 0
    assert span.attributes["gen_ai.usage.output_tokens"] == 2


async def test_a_stream_closed_early_still_ends_its_span_without_an_error(spans):
    deltas = [LLMStreamDelta("Hola"), LLMStreamDelta("", final=_response(SONNET_4_5, 40, 2))]
    provider = TrackedProvider(_Bedrock(deltas=deltas), "answer_direct")

    stream = provider.generate_stream(
        system="s", messages=[{"role": "user", "content": "q"}], model=SONNET_4_5
    )
    await anext(stream)
    await stream.aclose()

    (span,) = spans.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET
    assert "gen_ai.usage.input_tokens" not in span.attributes


async def test_an_embedding_request_is_a_span_priced_on_its_input(spans):
    created = []

    class _Embeddings:
        async def create(self, model, input):
            created.append(model)
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[3.0, 4.0]) for _ in input],
                usage=SimpleNamespace(prompt_tokens=50_000, total_tokens=50_000),
            )

    provider = OpenAIEmbeddingProvider(api_key="test", model="text-embedding-3-small")
    provider.client = SimpleNamespace(embeddings=_Embeddings())  # type: ignore[assignment]

    await provider.embed_batch(["a", "b"])

    (span,) = spans.get_finished_spans()
    assert span.name == "embeddings text-embedding-3-small"
    assert span.kind == SpanKind.CLIENT
    assert span.attributes["gen_ai.operation.name"] == "embeddings"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.usage.input_tokens"] == 50_000
    assert "gen_ai.usage.output_tokens" not in span.attributes
    assert span.attributes["gen_ai.usage.cost"] == 0.001


# ---------- a turn ----------


async def test_a_turn_sums_its_calls_and_puts_its_ids_in_the_logs(spans):
    classifier = TrackedProvider(_Bedrock(_response(GEMINI, 300, 20)), "classifier")
    generation = TrackedProvider(_Bedrock(_response(SONNET_4_5, 3446, 406)), "rag_generation")

    with workflow(AGENT_WORKFLOW, conversation_id="t1", channel="web", collection_id=4) as turn:
        await _call(classifier, GEMINI)
        await _call(generation, SONNET_4_5)
        log_event = add_trace_context(None, "info", {})
        log_context = structlog.contextvars.get_contextvars()
        turn.outcome("search_docs", None)

    *calls, root = spans.get_finished_spans()
    assert root.name == "invoke_workflow whisperdocs-agent"
    assert root.kind == SpanKind.INTERNAL
    assert root.parent is None
    assert all(c.parent.span_id == root.context.span_id for c in calls)
    assert {c.attributes["gen_ai.conversation.id"] for c in calls} == {"t1"}
    assert root.attributes["gen_ai.operation.name"] == "invoke_workflow"
    assert root.attributes["gen_ai.agent.name"] == "whisperdocs-agent"
    assert root.attributes["session.id"] == "t1"
    assert root.attributes["gen_ai.data_source.id"] == "4"
    assert root.attributes["whisperdocs.action"] == "search_docs"
    # The classifier's tokens and cost count too (audit A5).
    assert root.attributes["gen_ai.usage.input_tokens"] == 3746
    assert root.attributes["gen_ai.usage.output_tokens"] == 426
    assert root.attributes["whisperdocs.cost_usd"] == round(
        estimate_cost_usd(GEMINI, 300, 20) + estimate_cost_usd(SONNET_4_5, 3446, 406), 6
    )

    assert turn.trace_id == format(root.context.trace_id, "032x")
    assert log_event["trace_id"] == turn.trace_id
    assert log_context["thread_id"] == "t1"
    # Outside the turn, neither.
    assert add_trace_context(None, "info", {}) == {}
    assert "thread_id" not in structlog.contextvars.get_contextvars()


async def test_a_call_outside_a_turn_adds_to_no_turn(spans):
    provider = TrackedProvider(_Bedrock(_response(GEMINI, 10, 2)), "eval_judge")

    await _call(provider, GEMINI)

    (span,) = spans.get_finished_spans()
    assert span.parent is None
    assert "gen_ai.conversation.id" not in span.attributes


# ---------- the routes ----------


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
    def __init__(self, metadata: dict | None = None, error: Exception | None = None) -> None:
        self.metadata = metadata or {}
        self.error = error

    async def ainvoke(self, _input, config):
        if self.error is not None:
            raise self.error
        return {
            "messages": [HumanMessage(content="q"), AIMessage(content="a")],
            "pending_action": "search_docs",
            "retrieved_chunks": [],
            "metadata": self.metadata,
        }

    async def astream_events(self, _input, config, version):
        state = await self.ainvoke(_input, config)
        yield {"event": "on_chain_end", "name": "LangGraph", "data": {"output": state}}


@pytest.fixture
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


async def _post(path: str, agent: _FakeAgent) -> httpx.Response:
    app.state.agent = agent
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
            return await c.post(path, json={"message": "q"})
    finally:
        del app.state.agent


def _turn(spans):
    (turn,) = [s for s in spans.get_finished_spans() if s.name.startswith("invoke_workflow")]
    return turn


@pytest.mark.parametrize("path", ["/agent/chat", "/agent/chat/stream"])
async def test_the_query_log_row_carries_the_turns_trace_id(spans, db, path):
    r = await _post(path, _FakeAgent({"turn_failed": False}))

    assert r.status_code == 200
    root = _turn(spans)
    (log,) = [o for o in db.added if isinstance(o, QueryLog)]
    assert log.trace_id == format(root.context.trace_id, "032x")
    assert root.attributes["gen_ai.conversation.id"] == log.session_id
    assert root.attributes["whisperdocs.channel"] == "web"
    # FastAPI's span of the request is the trace's root.
    (request,) = [s for s in spans.get_finished_spans() if s.parent is None]
    assert request.kind == SpanKind.SERVER
    assert request.attributes["http.route"] == f"/api/v1{path}"
    assert root.parent.span_id == request.context.span_id


async def test_a_failed_answer_marks_the_turn_with_the_failing_calls_kind(spans, db):
    errors = [
        {"step": "classifier", "kind": "billing", "model": GEMINI},
        {"step": "generation", "kind": "timeout", "model": SONNET_4_5},
    ]
    r = await _post("/agent/chat", _FakeAgent({"turn_failed": True, "turn_errors": errors}))

    assert r.json()["failed"] is True
    root = _turn(spans)
    assert root.attributes["error.type"] == "timeout"
    assert root.status.status_code == StatusCode.ERROR


async def test_a_turn_that_raises_is_an_error_span(spans, db):
    throttled = ClientError({"Error": {"Code": "ThrottlingException"}}, "Converse")

    r = await _post("/agent/chat", _FakeAgent(error=throttled))

    assert r.status_code == 502
    root = _turn(spans)
    assert root.attributes["error.type"] == "rate_limit"
    assert not root.events


async def test_a_stream_that_fails_marks_its_turn(spans, db):
    r = await _post("/agent/chat/stream", _FakeAgent(error=RuntimeError("boom")))

    events = [block.split("\n", 1) for block in r.text.strip().split("\n\n")]
    (error,) = [data for name, data in events if name == "event: error"]
    assert json.loads(error.removeprefix("data: "))["code"] == "agent_failed"
    assert _turn(spans).status.status_code == StatusCode.ERROR


async def test_a_single_turn_answer_is_a_query_workflow_with_its_trace_id(spans, monkeypatch):
    from app.services import answering

    chunk = RetrievedChunk(
        chunk_id=1,
        document_id=1,
        document_title="d.md",
        chunk_index=0,
        text="t",
        relevance=0.9,
        cosine=0.9,
    )

    async def fake_retrieve(**_kwargs):
        return [chunk]

    class _Embedder:
        async def embed(self, _text):
            return [0.0] * 4

    provider = TrackedProvider(_Bedrock(_response(SONNET_4_5, 100, 20)), "rag_generation")
    monkeypatch.setattr(answering, "retrieve_pipeline", fake_retrieve)
    monkeypatch.setattr(answering, "llm_service_for", lambda _node: (provider, SONNET_4_5, 0.3))
    db = _FakeDb()

    await answering.answer_question(
        db,  # type: ignore[arg-type]
        _Embedder(),  # type: ignore[arg-type]
        QueryRequest(question="q", session_id="s1"),
        channel="mcp",
    )

    chat, root = spans.get_finished_spans()
    assert root.name == "invoke_workflow whisperdocs-query"
    assert root.attributes["whisperdocs.channel"] == "mcp"
    assert chat.parent.span_id == root.context.span_id
    (log,) = db.added
    assert log.trace_id == format(root.context.trace_id, "032x")


# ---------- the exporter ----------


def test_spans_are_exported_only_to_a_configured_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "otel_exporter_otlp_endpoint", "")
    provider = build_tracer_provider()
    assert provider._active_span_processor._span_processors == ()
    provider.shutdown()

    monkeypatch.setattr(
        settings, "otel_exporter_otlp_endpoint", "http://langfuse-web:3000/api/public/otel/"
    )
    monkeypatch.setattr(
        settings,
        "otel_exporter_otlp_headers",
        SecretStr("Authorization=Basic%20dGVzdDp0ZXN0,x-langfuse-ingestion-version=4"),
    )
    provider = build_tracer_provider()
    try:
        (processor,) = provider._active_span_processor._span_processors
        exporter = processor._batch_processor._exporter
        assert exporter._endpoint == "http://langfuse-web:3000/api/public/otel/v1/traces"
        headers = exporter._client._headers
        assert headers["authorization"] == "Basic dGVzdDp0ZXN0"
        assert headers["x-langfuse-ingestion-version"] == "4"
    finally:
        provider.shutdown()
