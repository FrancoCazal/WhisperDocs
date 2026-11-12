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


class _FakeModel:
    """Returns the logits it was given, one per pair, in order."""

    def __init__(self, logits: list[float]) -> None:
        self.logits = logits

    def predict(self, pairs):
        assert len(pairs) == len(self.logits)
        return self.logits


def _reranker(logits: list[float], threshold: float = 0.0) -> LocalCrossEncoderReranker:
    reranker = LocalCrossEncoderReranker(score_threshold=threshold)
    reranker._model = _FakeModel(logits)
    return reranker


@pytest.mark.asyncio
async def test_local_reranker_reorders_by_score():
    chunks = [_chunk(i, f"text {i}") for i in range(5)]

    out = await _reranker([float(i) for i in range(5)]).rerank("query", chunks, top_k=3)

    # Highest score wins → chunk_id 4, 3, 2.
    assert [c.chunk_id for c in out] == [4, 3, 2]


@pytest.mark.asyncio
async def test_the_score_is_the_sigmoid_of_the_logit_not_its_place_in_the_batch():
    """Min-max over the batch put the best chunk at 1.0 and the worst at 0.0
    even when none answered the question (audit R16)."""
    chunks = [_chunk(i, f"text {i}") for i in range(3)]

    relevant = await _reranker([4.0, 0.0, -4.0]).rerank("q", chunks, top_k=3)
    irrelevant = await _reranker([-6.0, -7.0, -8.0]).rerank("q", chunks, top_k=3)

    assert [c.relevance for c in relevant] == [0.982, 0.5, 0.018]
    # The best of a batch where nothing answers still scores low.
    assert irrelevant[0].relevance == 0.0025
    assert all(c.relevance < 0.01 for c in irrelevant)


@pytest.mark.asyncio
async def test_extreme_logits_do_not_overflow():
    chunks = [_chunk(0, "a"), _chunk(1, "b")]

    out = await _reranker([1000.0, -1000.0]).rerank("q", chunks, top_k=2)

    assert [c.relevance for c in out] == [1.0, 0.0]


@pytest.mark.asyncio
async def test_the_threshold_drops_weak_chunks_but_never_the_best():
    """A threshold that could leave a turn with no chunks would be the
    relevance gate, out of V1's scope."""
    chunks = [_chunk(i, f"text {i}") for i in range(3)]

    some = await _reranker([2.0, -1.0, 0.5], threshold=0.6).rerank("q", chunks, top_k=3)
    none_pass = await _reranker([-5.0, -6.0], threshold=0.6).rerank("q", chunks[:2], top_k=2)

    assert [c.chunk_id for c in some] == [0, 2]
    assert [c.chunk_id for c in none_pass] == [0]


@pytest.mark.asyncio
async def test_local_reranker_handles_tied_scores():
    chunks = [_chunk(i, f"text {i}") for i in range(3)]

    out = await _reranker([0.0] * 3).rerank("query", chunks, top_k=2)

    assert len(out) == 2
    assert all(c.relevance == 0.5 for c in out)


def test_the_model_is_asked_for_its_raw_logit(monkeypatch):
    """So the sigmoid is applied once, whatever activation a model's config
    sets: bge-reranker-base's would apply one, and min-max hid that."""
    import sys
    import types

    built: list[dict] = []

    class _CrossEncoder:
        def __init__(self, name, **kwargs):
            built.append({"name": name, **kwargs})

    class _Identity:
        pass

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=_CrossEncoder)
    )
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(nn=types.SimpleNamespace(Identity=_Identity))
    )

    reranker = LocalCrossEncoderReranker(model_name="some/reranker")
    model = reranker.load()

    assert reranker.load() is model
    assert len(built) == 1 and built[0]["name"] == "some/reranker"
    assert isinstance(built[0]["activation_fn"], _Identity)
