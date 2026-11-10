"""The record of each role's last calls (BE-14): kept by real calls, read
without calling anything."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.config import settings
from app.db import get_db
from app.main import app
from app.services import llm
from app.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.services.llm.base import LLMResponse, LLMStreamDelta
from app.services.model_status import ModelStatus, model_status

T0 = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)


class _Refused(Exception):
    """Vertex's refusal on a project without billing: a 403 that says so."""

    code = 403

    def __init__(self) -> None:
        super().__init__("This API method requires billing to be enabled")


class _Down(Exception):
    status_code = 503


RESPONSE = LLMResponse(text="a", tokens_input=1, tokens_output=1, model="m", latency_ms=1)


class _Provider:
    def __init__(self, fail: Exception | None = None, stream_fails_after: int | None = None):
        self.fail = fail
        self.stream_fails_after = stream_fails_after

    async def generate(self, **_):
        if self.fail:
            raise self.fail
        return RESPONSE

    async def generate_structured(self, **_):
        if self.fail:
            raise self.fail
        return object(), RESPONSE

    async def generate_stream(self, **_):
        for i in range(3):
            if self.stream_fails_after is not None and i == self.stream_fails_after:
                raise _Down()
            yield LLMStreamDelta(text_delta=f"t{i}")
        yield LLMStreamDelta(text_delta="", final=RESPONSE)


@pytest.fixture
def provider(monkeypatch):
    """What `llm_service_for` wraps, in place of Gemini or Bedrock."""
    holder = _Provider()
    monkeypatch.setattr(llm, "llm_provider_for_model", lambda _model: holder)
    return holder


KWARGS = {"system": "s", "messages": [], "temperature": 0.0, "max_tokens": 10}


def test_a_role_goes_from_unknown_to_failing_and_back_to_ok():
    now = [T0]
    status = ModelStatus(clock=lambda: now[0])
    model = settings.agent.model_classifier

    assert status.role("classifier")["state"] == "unknown"
    status.ok("classifier", model)
    now[0] = T0 + timedelta(minutes=1)
    assert status.failed("classifier", model, _Refused()) == "billing"
    now[0] = T0 + timedelta(minutes=2)
    status.failed("classifier", model, _Refused())

    failing = status.role("classifier")
    assert failing == {
        "model": model,
        "state": "failing",
        "calls": 3,
        "failures_in_a_row": 2,
        "last_ok_at": T0,
        "last_failure_at": T0 + timedelta(minutes=2),
        "last_failure_kind": "billing",
    }
    status.ok("classifier", model)
    recovered = status.role("classifier")
    assert (recovered["state"], recovered["failures_in_a_row"]) == ("ok", 0)
    # The last failure stays on record.
    assert recovered["last_failure_kind"] == "billing"


def test_a_role_reports_the_model_it_is_configured_with(monkeypatch):
    status = ModelStatus()
    status.failed("classifier", settings.agent.model_classifier, _Refused())
    monkeypatch.setattr(settings.agent, "model_classifier", "gemini-other")

    assert status.role("classifier") == {
        "model": "gemini-other",
        "state": "unknown",
        "calls": 0,
        "failures_in_a_row": 0,
        "last_ok_at": None,
        "last_failure_at": None,
        "last_failure_kind": None,
    }


async def test_every_call_through_llm_service_for_is_recorded_under_its_role(provider):
    classifier, model, _ = llm.llm_service_for("classifier")
    await classifier.generate_structured(model=model, response_model=object, **KWARGS)
    provider.fail = _Refused()
    rag, rag_model, _ = llm.llm_service_for("rag")
    with pytest.raises(_Refused):
        await rag.generate(model=rag_model, **KWARGS)

    snapshot = model_status.snapshot()
    assert snapshot["classifier"]["state"] == "ok"
    # "rag" is the alias /query and search_docs use: its role is rag_generation.
    assert (
        snapshot["rag_generation"]["state"],
        snapshot["rag_generation"]["last_failure_kind"],
    ) == (
        "failing",
        "billing",
    )
    assert snapshot["summarize"]["state"] == "unknown"


async def test_a_stream_counts_once_it_ends_or_fails(provider):
    tracked, model, _ = llm.llm_service_for("answer_direct")
    deltas = [d async for d in tracked.generate_stream(model=model, **KWARGS)]
    assert deltas[-1].final is RESPONSE
    assert model_status.role("answer_direct")["calls"] == 1

    provider.stream_fails_after = 1
    with pytest.raises(_Down):
        async for _ in tracked.generate_stream(model=model, **KWARGS):
            pass
    role = model_status.role("answer_direct")
    assert (role["state"], role["calls"], role["last_failure_kind"]) == (
        "failing",
        2,
        "unavailable",
    )


async def test_a_stream_its_reader_leaves_counts_nothing(provider):
    """A client gone mid-answer says nothing about the model."""
    tracked, model, _ = llm.llm_service_for("answer_direct")
    stream = tracked.generate_stream(model=model, **KWARGS)
    await anext(stream)
    await stream.aclose()

    assert model_status.role("answer_direct")["calls"] == 0


class _Embeddings:
    def __init__(self, fail: Exception | None) -> None:
        self.fail = fail

    async def create(self, model, input):  # noqa: A002 - the SDK's name
        if self.fail:
            raise self.fail

        class _Item:
            embedding = [1.0, 0.0]

        class _Usage:
            prompt_tokens = total_tokens = 3

        class _Response:
            data = [_Item() for _ in input]
            usage = _Usage()

        return _Response()


async def test_the_embedder_records_its_calls():
    embedder = OpenAIEmbeddingProvider(api_key="test")
    embedder.client = type("Client", (), {"embeddings": _Embeddings(None)})()  # type: ignore[assignment]
    await embedder.embed("q")
    assert model_status.role("embedding")["state"] == "ok"

    embedder.client.embeddings.fail = _Down()  # type: ignore[attr-defined]
    with pytest.raises(Exception, match="Failed to generate embeddings"):
        await embedder.embed("q")
    role = model_status.role("embedding")
    assert (role["state"], role["last_failure_kind"], role["model"]) == (
        "failing",
        "unavailable",
        settings.embedding_model,
    )


async def _get(path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
        return await c.get(path, **kwargs)


async def test_the_status_route_reads_the_record():
    model_status.failed("classifier", settings.agent.model_classifier, _Refused())

    r = await _get("/admin/model-status")

    assert r.status_code == 200
    body = r.json()
    assert list(body["roles"]) == [
        "classifier",
        "query_rewrite",
        "rag_generation",
        "clarification",
        "answer_direct",
        "summarize",
        "eval_judge",
        "embedding",
    ]
    classifier = body["roles"]["classifier"]
    assert (classifier["state"], classifier["last_failure_kind"]) == ("failing", "billing")
    assert classifier["last_failure_at"].endswith("Z") or "+00:00" in classifier["last_failure_at"]
    assert body["roles"]["embedding"]["state"] == "unknown"


@pytest.fixture
def healthy_db_and_embedder(monkeypatch):
    class _Db:
        async def execute(self, _statement):
            class _Result:
                def first(self):
                    return ("vector",)

            return _Result()

    async def fake_db():
        yield _Db()

    class _Embedder:
        """Counts the embedding calls; fails them while `down`."""

        calls = 0
        down = False

        async def embed(self, _text):
            _Embedder.calls += 1
            if _Embedder.down:
                raise ConnectionRefusedError("embedder unreachable")
            return [1.0]

    app.dependency_overrides[get_db] = fake_db
    monkeypatch.setattr("app.api.v1.health.EmbeddingService", _Embedder)
    yield _Embedder
    app.dependency_overrides.clear()


async def test_a_deep_check_adds_the_roles_and_a_failing_one_degrades_it(healthy_db_and_embedder):
    ok = await _get("/health", params={"deep": "true"})
    assert ok.status_code == 200
    assert ok.json()["status"] == "ok"
    assert set(ok.json()["models"].values()) == {"unknown"}

    model_status.failed("classifier", settings.agent.model_classifier, _Refused())
    response = await _get("/health", params={"deep": "true"})
    degraded = response.json()

    # Degraded answers 503, with the body (audit E12).
    assert response.status_code == 503
    assert degraded["status"] == "degraded"
    assert degraded["checks"] == {"database": "ok", "pgvector": "ok", "embedder": "ok"}
    assert degraded["models"]["classifier"] == "failing: billing"
    # The probe without deep says nothing of the models, and its database is up.
    shallow = await _get("/health")
    assert shallow.status_code == 200 and "models" not in shallow.json()


async def test_the_deep_check_reuses_the_embedder_result(healthy_db_and_embedder, monkeypatch):
    """Each deep check paid an embedding call (audit E12): an "ok" is kept 5
    minutes, a failure 30 seconds."""
    embedder = healthy_db_and_embedder
    clock = [1000.0]
    monkeypatch.setattr("app.api.v1.health.time.monotonic", lambda: clock[0])

    async def embedder_check() -> str:
        return (await _get("/health", params={"deep": "true"})).json()["checks"]["embedder"]

    assert [await embedder_check(), await embedder_check()] == ["ok", "ok"]
    assert embedder.calls == 1

    embedder.down = True
    clock[0] += 299
    assert await embedder_check() == "ok"
    clock[0] += 2
    assert await embedder_check() == "error: unavailable"
    assert embedder.calls == 2

    embedder.down = False
    clock[0] += 29
    assert await embedder_check() == "error: unavailable"
    clock[0] += 2
    assert await embedder_check() == "ok"
    assert embedder.calls == 3
