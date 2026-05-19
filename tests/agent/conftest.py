"""Shared fixtures for agent tests.

Mocks the provider layer and retrieval so tests exercise agent logic
without live LLM calls, OpenAI embeddings, or Postgres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.agent.schemas import AgentAction
from app.services.llm.base import LLMResponse, LLMStreamDelta


@dataclass
class FakeProvider:
    """Minimal LLMProvider stand-in.

    `generate_response` is returned by `generate()`.
    `structured_response` is returned by `generate_structured()` — must be
    a tuple (parsed_model, LLMResponse).
    """

    generate_response: LLMResponse = field(
        default_factory=lambda: LLMResponse(
            text="fake answer",
            tokens_input=10,
            tokens_output=5,
            model="fake-model",
            latency_ms=100,
        )
    )
    structured_response: tuple[Any, LLMResponse] | None = None
    calls: list[dict] = field(default_factory=list)

    async def generate(self, **kwargs) -> LLMResponse:
        self.calls.append({"method": "generate", "kwargs": kwargs})
        return self.generate_response

    async def generate_structured(self, **kwargs) -> tuple[Any, LLMResponse]:
        self.calls.append({"method": "generate_structured", "kwargs": kwargs})
        if self.structured_response is None:
            raise RuntimeError("FakeProvider.structured_response not configured")
        return self.structured_response

    async def generate_stream(self, **kwargs):
        """Yields a single delta carrying the full response — sufficient for tests."""
        self.calls.append({"method": "generate_stream", "kwargs": kwargs})
        resp = self.generate_response
        if resp.text:
            yield LLMStreamDelta(text_delta=resp.text, final=None)
        yield LLMStreamDelta(text_delta="", final=resp)


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def mock_llm_service_for(monkeypatch):
    """Return a helper that stubs `llm_service_for` in a given module path.

    Usage:
        mock_llm_service_for(
            "app.agent.nodes.decide_action", fake_provider, model="m", temp=0.0
        )
    """

    def _install(module_path: str, provider, model: str = "fake-model", temp: float = 0.0):
        monkeypatch.setattr(
            f"{module_path}.llm_service_for",
            lambda _node: (provider, model, temp),
        )

    return _install


@pytest.fixture
def classifier_response():
    """Build a (AgentAction, LLMResponse) tuple for generate_structured."""

    def _build(
        action: str,
        reason: str = "test",
        confidence: str = "high",
    ):
        parsed = AgentAction(  # type: ignore[arg-type]
            action=action,
            reason=reason,
            confidence=confidence,
        )
        resp = LLMResponse(
            text="",
            tokens_input=50,
            tokens_output=20,
            model="fake-classifier",
            latency_ms=200,
        )
        return parsed, resp

    return _build


@pytest.fixture
def mock_embedder(monkeypatch):
    """Patch EmbeddingService.embed to return a deterministic vector."""
    from app.services import embedder as embedder_mod

    async def fake_embed(self, text):  # noqa: ARG001
        return [0.0] * 1536

    monkeypatch.setattr(embedder_mod.EmbeddingService, "embed", fake_embed)
    monkeypatch.setattr(embedder_mod.EmbeddingService, "__init__", lambda self, *a, **kw: None)


@pytest.fixture
def mock_retriever(monkeypatch):
    """Patch retrieval pipeline + VectorRetriever methods to return a configurable chunk list.

    Both the pipeline (used by search_docs_node, query.py, evaluator) and the raw
    retriever (legacy callers) are patched so tests can swap chunks regardless of
    which call path the code-under-test uses.
    """
    from app.services import retrieval_pipeline as pipeline_mod
    from app.services import retriever as retriever_mod

    state = {"chunks": []}

    async def fake_retrieve(*args, **kwargs):  # noqa: ARG001
        return list(state["chunks"])

    async def fake_method(self, *args, **kwargs):  # noqa: ARG001
        return list(state["chunks"])

    monkeypatch.setattr(pipeline_mod, "retrieve", fake_retrieve)
    # Re-bind in importer modules (they did `from ... import retrieve as retrieve_pipeline`).
    for module_path in (
        "app.agent.nodes.search_docs",
        "app.api.v1.query",
        "app.eval.evaluator",
    ):
        try:
            mod = __import__(module_path, fromlist=["retrieve_pipeline"])
        except ImportError:
            continue
        if hasattr(mod, "retrieve_pipeline"):
            monkeypatch.setattr(mod, "retrieve_pipeline", fake_retrieve)
    monkeypatch.setattr(retriever_mod.VectorRetriever, "retrieve", fake_method)
    monkeypatch.setattr(retriever_mod.VectorRetriever, "hybrid_retrieve", fake_method)

    def _set(chunks):
        state["chunks"] = chunks

    return _set


@pytest.fixture
def mock_session_local(monkeypatch):
    """Replace app.db.SessionLocal with a no-op async context manager so
    search_docs_node can enter `async with SessionLocal() as db` without Postgres.
    """

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    def _factory():
        return _FakeSession()

    monkeypatch.setattr("app.agent.nodes.search_docs.SessionLocal", _factory)


@pytest.fixture
def rag_chunk():
    from app.services.retriever import RetrievedChunk

    def _build(idx: int = 0, rel: float = 0.9):
        return RetrievedChunk(
            chunk_id=100 + idx,
            document_id=1,
            document_title="tino_lore.md",
            chunk_index=idx,
            text=f"chunk text {idx}",
            relevance=rel,
        )

    return _build
