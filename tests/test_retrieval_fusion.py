"""Hybrid fusion and score semantics (audit R1, R2, N3). No database."""

from __future__ import annotations

from app.services.fulltext import TS_CONFIGS, or_tsquery_sql
from app.services.mmr import mmr_select
from app.services.reranker import LocalCrossEncoderReranker
from app.services.retriever import RetrievedChunk, fuse_rrf


def _chunk(chunk_id: int, doc: int, relevance: float, cosine: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc,
        document_title=f"doc-{doc}.md",
        chunk_index=0,
        text=f"text {chunk_id}",
        relevance=relevance,
        cosine=cosine,
    )


def _vector(*ids_and_cos: tuple[int, float]) -> list[RetrievedChunk]:
    return [_chunk(cid, cid, cos, cos) for cid, cos in ids_and_cos]


def _lexical(*ids: int) -> list[RetrievedChunk]:
    return [_chunk(cid, cid, 1.0 - i * 0.1, None) for i, cid in enumerate(ids)]


def test_chunk_found_by_both_legs_ranks_first_with_relevance_one():
    vector = _vector((1, 0.62), (2, 0.60), (3, 0.55))
    lexical = _lexical(1, 4)

    out = fuse_rrf(vector, lexical, top_k=4)

    assert [c.chunk_id for c in out][0] == 1
    assert out[0].relevance == 1.0
    assert out[0].cosine == 0.62


def test_relevance_follows_the_order():
    vector = _vector((1, 0.40), (2, 0.70), (3, 0.65))
    lexical = _lexical(3, 1)

    out = fuse_rrf(vector, lexical, top_k=3)

    rel = [c.relevance for c in out]
    assert rel == sorted(rel, reverse=True)


def test_lexical_only_chunk_has_no_cosine():
    out = fuse_rrf(_vector((1, 0.5)), _lexical(9), top_k=2)

    by_id = {c.chunk_id: c for c in out}
    assert by_id[9].cosine is None
    assert by_id[1].cosine == 0.5


def test_empty_lexical_leg_keeps_vector_order_and_scale():
    vector = _vector((1, 0.7), (2, 0.6), (3, 0.5))

    out = fuse_rrf(vector, [], top_k=2)

    assert [c.chunk_id for c in out] == [1, 2]
    assert out[0].relevance == 1.0


def test_no_rows_in_either_leg():
    assert fuse_rrf([], [], top_k=5) == []


def test_or_tsquery_uses_every_config_and_no_web_syntax():
    sql = or_tsquery_sql("q")

    for cfg in TS_CONFIGS:
        assert f"plainto_tsquery('{cfg}'::regconfig, :q)" in sql
    assert "websearch_to_tsquery" not in sql
    assert "'&', '|'" in sql


def test_mmr_at_lambda_one_keeps_the_fused_order():
    fused = fuse_rrf(_vector((1, 0.4), (2, 0.7), (3, 0.6)), _lexical(3, 2), top_k=3)

    out = mmr_select(fused, top_k=2, lambda_=1.0)

    assert [c.chunk_id for c in out] == [c.chunk_id for c in fused[:2]]


async def test_reranker_keeps_cosine():
    reranker = LocalCrossEncoderReranker()

    class _FakeModel:
        def predict(self, pairs):
            return [float(i) for i in range(len(pairs))]

    reranker._model = _FakeModel()
    chunks = [_chunk(1, 1, 1.0, 0.61), _chunk(2, 2, 0.8, None)]

    out = await reranker.rerank("q", chunks, top_k=2)

    assert [c.chunk_id for c in out] == [2, 1]
    assert {c.chunk_id: c.cosine for c in out} == {1: 0.61, 2: None}
