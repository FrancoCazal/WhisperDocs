"""The API key (audit R6), the error bodies (R9, A9) and the rate limits (R9).

Runs the real app over ASGI, without its lifespan, with the database, the
agent and the providers replaced.
"""

from __future__ import annotations

import json
import re

import httpx
import pytest
from pydantic import SecretStr

from app.api.rate_limit import WINDOW_S, RateLimiter
from app.config import settings
from app.db import get_db
from app.main import app
from app.services.embedder import EmbeddingError, get_embedder
from app.services.ingestion import job_error_message
from app.services.retriever import RetrievedChunk
from app.services.text_extractor import UnreadableDocumentError

KEY = "test"
# What a leaked exception would carry: none of it may reach a response.
SECRET = "SELECT password FROM users"


async def _call(method: str, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
        return await c.request(method, path, **kwargs)


class _Db:
    """A session whose every statement succeeds, or raises `fail`."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail

    async def execute(self, _statement, *_a, **_kw):
        if self.fail is not None:
            raise self.fail

        class _Result:
            def first(self):
                return ("vector",)

        return _Result()


@pytest.fixture
def db():
    holder = _Db()

    async def fake_db():
        yield holder

    app.dependency_overrides[get_db] = fake_db
    yield holder
    app.dependency_overrides.clear()


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setattr(settings, "api_key", SecretStr(KEY))


# ---- R6: the key ----


async def test_without_api_key_set_the_api_is_open():
    r = await _call("GET", "/admin/settings")
    assert r.status_code == 200


async def test_an_empty_api_key_leaves_it_open(monkeypatch):
    monkeypatch.setattr(settings, "api_key", SecretStr(""))
    assert (await _call("GET", "/admin/settings")).status_code == 200


async def test_with_api_key_set_a_request_needs_it(with_key):
    missing = await _call("GET", "/admin/settings")
    wrong = await _call("GET", "/admin/settings", headers={"X-API-Key": "nope"})
    right = await _call("GET", "/admin/settings", headers={"X-API-Key": KEY})

    assert missing.status_code == 401
    assert missing.json()["code"] == "api_key_required"
    assert missing.headers["www-authenticate"] == "ApiKey"
    assert wrong.status_code == 401
    assert wrong.json()["code"] == "api_key_invalid"
    assert right.status_code == 200


async def test_a_probe_needs_no_key_but_a_deep_check_does(with_key, db):
    assert (await _call("GET", "/health")).status_code == 200
    deep = await _call("GET", "/health", params={"deep": "true"})
    assert deep.status_code == 401
    assert deep.json()["code"] == "api_key_required"


async def test_the_whatsapp_webhook_checks_twilios_signature_not_the_key(with_key, db):
    r = await _call("POST", "/webhook/whatsapp", data={"From": "whatsapp:+1", "Body": "hi"})
    assert r.status_code == 403
    assert r.json()["code"] == "forbidden"


async def test_every_route_but_the_probe_and_the_webhooks_needs_the_key(with_key):
    """A router added without the key fails here, not in production. The key
    is checked before the route's parameters, so a bare request gets the 401."""
    open_paths = {"/api/v1/health", "/api/v1/webhook/whatsapp", "/api/v1/webhook/whatsapp/status"}
    operations = [
        (method, re.sub(r"\{[^}]+\}", "1", path).removeprefix("/api/v1"))
        for path, ops in app.openapi()["paths"].items()
        if path not in open_paths
        for method in ops
    ]

    statuses = {f"{m.upper()} {p}": (await _call(m, p)).status_code for m, p in operations}

    assert len(statuses) > 30
    assert {op: s for op, s in statuses.items() if s != 401} == {}


def test_the_openapi_schema_offers_the_key():
    schemes = app.openapi()["components"]["securitySchemes"]
    assert schemes["APIKeyHeader"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "Needed when the API runs with API_KEY set.",
    }


# ---- R9 and A9: error bodies ----


async def test_an_error_has_a_code_and_the_request_id_it_returns():
    r = await _call("GET", "/no-such-route")

    assert r.status_code == 404
    body = r.json()
    assert body["code"] == "not_found"
    assert body["request_id"] == r.headers["x-request-id"]


async def test_a_plain_request_id_is_kept_and_any_other_replaced():
    kept = await _call("GET", "/no-such-route", headers={"X-Request-ID": "abc-123"})
    replaced = await _call("GET", "/no-such-route", headers={"X-Request-ID": "a b\nc"})

    assert kept.headers["x-request-id"] == "abc-123"
    assert replaced.headers["x-request-id"] not in ("a b\nc", "")


async def test_a_validation_error_keeps_fastapis_detail_and_adds_a_code():
    r = await _call("POST", "/agent/chat", json={"message": ""})

    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"
    assert r.json()["detail"][0]["loc"] == ["body", "message"]


async def test_an_unhandled_error_is_a_500_without_its_text_and_with_cors():
    async def broken_db():
        raise RuntimeError(SECRET)
        yield  # pragma: no cover

    app.dependency_overrides[get_db] = broken_db
    try:
        r = await _call("GET", "/notes", headers={"Origin": "http://localhost:5173"})
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 500
    assert r.json()["code"] == "internal_error"
    assert SECRET not in r.text
    assert r.json()["detail"] == f"Internal error. Request id {r.headers['x-request-id']}."
    # From inside CORS: the browser sees the 500, not a CORS error.
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "X-Request-ID" in r.headers["access-control-expose-headers"]


class _FailingAgent:
    async def ainvoke(self, _input, config):
        raise TimeoutError(SECRET)

    async def astream_events(self, _input, config, version):
        raise TimeoutError(SECRET)
        yield  # pragma: no cover


@pytest.fixture
def failing_agent(monkeypatch):
    class _Conversations:
        """The stream's own session: no thread yet, so it creates one."""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def scalar(self, _statement):
            return None

        def add(self, _obj) -> None:
            pass

        async def flush(self) -> None:
            pass

        async def commit(self) -> None:
            pass

    async def fake_db():
        yield _Conversations()

    monkeypatch.setattr("app.api.v1.agent.SessionLocal", _Conversations)
    app.dependency_overrides[get_db] = fake_db
    app.state.agent = _FailingAgent()
    yield
    del app.state.agent
    app.dependency_overrides.clear()


async def test_a_failed_agent_says_why_without_the_exceptions_text(failing_agent):
    r = await _call("POST", "/agent/chat", json={"message": "q"})

    assert r.status_code == 502
    body = r.json()
    assert (body["code"], body["kind"]) == ("agent_failed", "timeout")
    assert SECRET not in r.text
    assert body["detail"].startswith("The agent failed: it timed out. Request id ")


async def test_a_failed_stream_says_why_without_the_exceptions_text(failing_agent):
    r = await _call("POST", "/agent/chat/stream", json={"message": "q"})

    assert SECRET not in r.text
    (error,) = [
        json.loads(block.split("data: ", 1)[1])
        for block in r.text.strip().split("\n\n")
        if block.startswith("event: error")
    ]
    assert (error["code"], error["kind"]) == ("agent_failed", "timeout")
    assert error["request_id"] == r.headers["x-request-id"]
    assert error["message"].startswith("The agent failed: it timed out.")


async def test_a_failed_query_embedding_is_a_502_with_its_kind(db):
    class _Embedder:
        async def embed(self, _text):
            raise EmbeddingError(f"Failed to generate embeddings: {SECRET}") from TimeoutError()

    app.dependency_overrides[get_embedder] = lambda: _Embedder()
    r = await _call("POST", "/query", json={"question": "q"})

    assert r.status_code == 502
    assert (r.json()["code"], r.json()["kind"]) == ("upstream_error", "timeout")
    assert SECRET not in r.text


async def test_a_failed_query_generation_is_a_502_not_a_bare_500(db, monkeypatch):
    """The providers raise their SDK's exceptions; `except LLMError` let them
    through as a 500."""

    class _Embedder:
        async def embed(self, _text):
            return [0.0] * 3

    class _Provider:
        async def generate(self, **_):
            raise ConnectionError(SECRET)

    async def retrieve(**_):
        chunk = RetrievedChunk(
            chunk_id=1,
            document_id=1,
            document_title="doc.md",
            chunk_index=0,
            text="chunk text",
            relevance=0.9,
            cosine=0.9,
        )
        return [chunk]

    app.dependency_overrides[get_embedder] = lambda: _Embedder()
    monkeypatch.setattr("app.api.v1.query.retrieve_pipeline", retrieve)
    monkeypatch.setattr("app.api.v1.query.llm_service_for", lambda _n: (_Provider(), "m", 0.0))
    r = await _call("POST", "/query", json={"question": "q"})

    assert r.status_code == 502
    assert (r.json()["code"], r.json()["kind"]) == ("upstream_error", "unavailable")
    assert SECRET not in r.text


async def test_a_failed_health_check_says_its_kind_not_its_text(db):
    db.fail = ConnectionRefusedError(SECRET)
    r = await _call("GET", "/health")

    assert r.json() == {"status": "degraded", "checks": {"database": "error: unavailable"}}


def test_a_failed_jobs_message_keeps_rejections_and_drops_internals():
    assert job_error_message(UnreadableDocumentError("Could not read the PDF")) == (
        "Could not read the PDF"
    )
    embedding = EmbeddingError(SECRET)
    embedding.__cause__ = TimeoutError()
    assert job_error_message(embedding) == "Embedding the document failed: it timed out."
    assert SECRET not in job_error_message(RuntimeError(SECRET))


async def test_a_body_over_the_limit_is_a_413_with_a_code():
    r = await _call(
        "POST",
        "/documents/upload",
        content=b"x",
        headers={"Content-Length": str(100 * 1024 * 1024), "Content-Type": "multipart/form-data"},
    )

    assert r.status_code == 413
    assert r.json()["code"] == "payload_too_large"
    assert r.json()["request_id"] == r.headers["x-request-id"]


# ---- R9: rate limits ----


def test_the_window_lets_a_request_in_once_the_oldest_leaves_it():
    now = [0.0]
    limiter = RateLimiter(clock=lambda: now[0])

    assert limiter.hit("b", "c", 2) is None
    now[0] = 10.0
    assert limiter.hit("b", "c", 2) is None
    now[0] = 20.0
    assert limiter.hit("b", "c", 2) == pytest.approx(WINDOW_S - 20.0)
    # Another client, or another bucket, counts on its own.
    assert limiter.hit("b", "other", 2) is None
    assert limiter.hit("other", "c", 2) is None
    now[0] = WINDOW_S
    assert limiter.hit("b", "c", 2) is None


class _Agent:
    async def ainvoke(self, _input, config):
        from langchain_core.messages import AIMessage, HumanMessage

        return {
            "messages": [HumanMessage(content="q"), AIMessage(content="a")],
            "pending_action": "answer_direct",
            "retrieved_chunks": [],
            "metadata": {},
        }


@pytest.fixture
def agent():
    class _Conversations:
        async def scalar(self, _statement):
            return None

        def add(self, _obj) -> None:
            pass

        async def flush(self) -> None:
            pass

        async def commit(self) -> None:
            pass

    async def fake_db():
        yield _Conversations()

    app.dependency_overrides[get_db] = fake_db
    app.state.agent = _Agent()
    yield
    del app.state.agent
    app.dependency_overrides.clear()


async def test_past_the_limit_a_question_gets_a_429_with_retry_after(agent, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)

    statuses = [(await _call("POST", "/agent/chat", json={"message": "q"})) for _ in range(3)]

    assert [r.status_code for r in statuses] == [200, 200, 429]
    limited = statuses[-1]
    assert limited.json()["code"] == "rate_limited"
    assert 1 <= int(limited.headers["retry-after"]) <= WINDOW_S


async def test_with_auth_off_a_made_up_key_is_no_new_client(agent, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)

    first = await _call("POST", "/agent/chat", json={"message": "q"}, headers={"X-API-Key": "a"})
    second = await _call("POST", "/agent/chat", json={"message": "q"}, headers={"X-API-Key": "b"})

    assert (first.status_code, second.status_code) == (200, 429)


async def test_with_auth_on_each_key_has_its_own_window(agent, monkeypatch, with_key):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
    headers = {"X-API-Key": KEY}

    assert (
        await _call("POST", "/agent/chat", json={"message": "q"}, headers=headers)
    ).status_code == 200
    assert (
        await _call("POST", "/agent/chat", json={"message": "q"}, headers=headers)
    ).status_code == 429
    # A wrong key is refused before it is counted.
    wrong = await _call("POST", "/agent/chat", json={"message": "q"}, headers={"X-API-Key": "x"})
    assert wrong.json()["code"] == "api_key_invalid"


async def test_a_limit_of_zero_is_off(agent):
    statuses = {
        (await _call("POST", "/agent/chat", json={"message": "q"})).status_code for _ in range(5)
    }
    assert statuses == {200}


async def test_an_eval_run_has_its_own_limit_and_a_dataset_cap(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_eval_per_minute", 1)
    # The route's dependencies run before its body is validated: without
    # OPENAI_API_KEY, the real embedder would fail first.
    app.dependency_overrides[get_embedder] = lambda: object()
    # Both fail validation, after the limit counted them.
    item = {"question": "q", "expected_answer": "a"}
    try:
        too_many = await _call("POST", "/eval/run", json={"dataset": [item] * 201})
        limited = await _call("POST", "/eval/run", json={"dataset": []})
    finally:
        app.dependency_overrides.clear()

    assert too_many.status_code == 422
    assert limited.status_code == 429
