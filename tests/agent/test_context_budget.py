"""One candidate pool and one context budget for every path (audit R15).

The reranker pool was capped at 20 rows per leg whatever
RERANKER_CANDIDATE_POOL said, and `/query` and the eval sent every retrieved
chunk while the chat applied `rag_context_token_budget`. The eval therefore
did not measure the prompt the chat sends.
"""

from __future__ import annotations

import httpx
import pytest

from app.config import settings
from app.eval.evaluator import RAGEvaluator
from app.eval import schemas as eval_schemas
from app.eval.schemas import Category, Difficulty, JudgeResult
from app.services import retrieval_pipeline
from app.services.rag import build_user_prompt
from app.services.retriever import RetrievedChunk

# About 600 tokens each: 20 of them do not fit a 6000-token budget.
_LONG = "Support tickets are answered within one business day by the on-call team. " * 40


def _chunk(idx: int, text: str = _LONG) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=idx,
        document_id=1,
        document_title=f"doc{idx}.md",
        chunk_index=idx,
        text=text,
        relevance=1.0 - idx / 100,
        cosine=0.9,
    )


# --- candidate pool -------------------------------------------------------------


class _RecordingRetriever:
    calls: list[dict] = []

    def __init__(self, _db) -> None:
        pass

    async def hybrid_retrieve(self, **kwargs):
        _RecordingRetriever.calls.append(kwargs)
        return [_chunk(i, "short") for i in range(kwargs["top_k"])]


class _KeepOrderReranker:
    received: int = 0

    async def rerank(self, query, chunks, top_k):
        _KeepOrderReranker.received = len(chunks)
        return chunks[:top_k]


@pytest.fixture
def recording_retriever(monkeypatch):
    _RecordingRetriever.calls = []
    monkeypatch.setattr(retrieval_pipeline, "VectorRetriever", _RecordingRetriever)
    monkeypatch.setattr(settings, "mmr_enabled", False)
    return _RecordingRetriever.calls


async def test_the_reranker_gets_the_configured_pool(recording_retriever, monkeypatch):
    monkeypatch.setattr(settings, "reranker_candidate_pool", 50)

    chunks = await retrieval_pipeline.retrieve(
        db=None,
        query_text="q",
        query_embedding=[0.0],
        collection_id=1,
        top_k=5,
        reranker=_KeepOrderReranker(),
    )

    assert recording_retriever[-1]["top_k"] == 50
    assert recording_retriever[-1]["candidate_pool"] == 50
    assert _KeepOrderReranker.received == 50
    assert len(chunks) == 5


async def test_without_refinement_each_leg_still_returns_twenty(recording_retriever):
    """RRF needs depth: the default path keeps 20 rows per leg, as before."""
    from app.services.reranker import NoopReranker

    await retrieval_pipeline.retrieve(
        db=None,
        query_text="q",
        query_embedding=[0.0],
        collection_id=1,
        top_k=5,
        reranker=NoopReranker(),
    )

    assert recording_retriever[-1]["top_k"] == 5
    assert recording_retriever[-1]["candidate_pool"] == 20


# --- context budget --------------------------------------------------------------


def test_prompt_keeps_only_the_chunks_that_fit_the_budget():
    chunks = [_chunk(i) for i in range(20)]

    prompt, included = build_user_prompt("When are tickets answered?", chunks)

    assert 0 < len(included) < 20
    assert included == chunks[: len(included)]  # same size, so the best ones
    assert f"[{len(included)}]" in prompt
    assert f"[{len(included) + 1}]" not in prompt
    assert prompt.endswith("Question: When are tickets answered?\n\nAnswer:")


def test_prompt_without_chunks_says_so():
    prompt, included = build_user_prompt("q", [])
    assert included == []
    assert "(no context available)" in prompt


class _FakeDb:
    def add(self, _obj) -> None:
        pass

    async def commit(self) -> None:
        pass


class _FakeEmbedder:
    async def embed(self, _text: str) -> list[float]:
        return [0.0]


async def test_query_cites_only_the_chunks_in_the_prompt(
    fake_provider, mock_llm_service_for, monkeypatch
):
    from app.api.v1 import query as query_module
    from app.db import get_db
    from app.main import app
    from app.services.embedder import get_embedder

    chunks = [_chunk(i) for i in range(20)]

    async def fake_retrieve(**_kwargs):
        return chunks

    async def fake_db():
        yield _FakeDb()

    monkeypatch.setattr(query_module, "retrieve_pipeline", fake_retrieve)
    mock_llm_service_for("app.api.v1.query", fake_provider)
    app.dependency_overrides[get_db] = fake_db
    app.dependency_overrides[get_embedder] = lambda: _FakeEmbedder()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/api/v1/query", json={"question": "q", "top_k": 20})
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200
    _, included = build_user_prompt("q", chunks)
    sent = fake_provider.calls[-1]["kwargs"]["messages"][-1]["content"]
    assert f"[{len(included)}]" in sent and f"[{len(included) + 1}]" not in sent
    assert [c["chunk_id"] for c in r.json()["citations"]] == [c.chunk_id for c in included]
    assert r.json()["metadata"]["chunks_retrieved"] == 20


class _RecordingJudge:
    def __init__(self) -> None:
        self.contexts: list[str] = []

    async def judge_correctness(self, _item, _answer) -> JudgeResult:
        return JudgeResult(passed=True, reasoning="ok")

    async def judge_hallucination(self, _answer, contexts) -> JudgeResult:
        self.contexts = contexts
        return JudgeResult(passed=True, reasoning="ok")


class _NoSession:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *_exc):
        return False


async def test_eval_judges_the_chunks_the_model_saw(fake_provider, monkeypatch):
    from app.eval import evaluator as evaluator_module

    chunks = [_chunk(i) for i in range(20)]

    async def fake_retrieve(**_kwargs):
        return chunks

    monkeypatch.setattr(evaluator_module, "retrieve_pipeline", fake_retrieve)
    monkeypatch.setattr(settings.agent, "rag_structured_output", False)
    monkeypatch.setattr(settings.agent, "citation_semantic_validation", False)
    judge = _RecordingJudge()
    evaluator = RAGEvaluator(
        session_maker=_NoSession,
        embedder=_FakeEmbedder(),
        provider=fake_provider,
        model="fake-model",
        temperature=0.0,
        judge=judge,
        collection_id=1,
        top_k=20,
    )
    item = eval_schemas.TestItem(
        id=1,
        question="q",
        expected_answer="a",
        difficulty=Difficulty.EASY,
        category=Category.IN_SCOPE,
        expected_sources=["doc0.md"],
    )

    result = await evaluator._evaluate_item(item)

    _, included = build_user_prompt("q", chunks)
    assert len(judge.contexts) == len(included) < 20
    # Retrieval metrics still score everything retrieval returned.
    assert len(result.citations) == 20
