import pytest

from app.services.reranker import LocalCrossEncoderReranker, NoopReranker
from app.services.retriever import RetrievedChunk


def _chunk(chunk_id: int, text: str, relevance: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id // 2,
        document_title=f"doc-{chunk_id // 2}",
        chunk_index=chunk_id,
        text=text,
        relevance=relevance,
    )


@pytest.mark.asyncio
async def test_noop_truncates_to_top_k():
    reranker = NoopReranker()
    chunks = [_chunk(i, f"text {i}") for i in range(10)]
    out = await reranker.rerank("query", chunks, top_k=3)
    assert [c.chunk_id for c in out] == [0, 1, 2]


@pytest.mark.asyncio
async def test_noop_handles_empty():
    out = await NoopReranker().rerank("q", [], top_k=5)
    assert out == []


@pytest.mark.asyncio
async def test_local_reranker_reorders_by_score(monkeypatch):
    """Bypass model load; inject scores directly to test ordering + normalization."""
    reranker = LocalCrossEncoderReranker()

    class _FakeModel:
        def predict(self, pairs):
            # First chunk gets lowest score; last gets highest. Expect reversal.
            return [float(i) for i in range(len(pairs))]

    monkeypatch.setattr(reranker, "_load", lambda: _FakeModel())

    chunks = [_chunk(i, f"text {i}") for i in range(5)]
    out = await reranker.rerank("query", chunks, top_k=3)
    # Highest score wins → chunk_id 4, 3, 2.
    assert [c.chunk_id for c in out] == [4, 3, 2]
    # Top result normalized to 1.0.
    assert out[0].relevance == 1.0
    # Last result normalized strictly less than 1.0.
    assert out[-1].relevance < 1.0


@pytest.mark.asyncio
async def test_local_reranker_handles_tied_scores(monkeypatch):
    reranker = LocalCrossEncoderReranker()

    class _FakeModel:
        def predict(self, pairs):
            return [0.5] * len(pairs)

    monkeypatch.setattr(reranker, "_load", lambda: _FakeModel())
    chunks = [_chunk(i, f"text {i}") for i in range(3)]
    out = await reranker.rerank("query", chunks, top_k=2)
    assert len(out) == 2
    # All tied → relevance falls back to 1.0 (zero span).
    assert all(c.relevance == 1.0 for c in out)
