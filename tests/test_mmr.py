from app.services.mmr import mmr_select
from app.services.retriever import RetrievedChunk


def _chunk(chunk_id: int, doc_id: int, relevance: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        document_title=f"doc-{doc_id}",
        chunk_index=chunk_id,
        text=f"chunk {chunk_id}",
        relevance=relevance,
    )


def test_returns_empty_for_no_candidates():
    assert mmr_select([], top_k=5) == []


def test_returns_all_when_pool_smaller_than_top_k():
    chunks = [_chunk(i, doc_id=i, relevance=0.5) for i in range(3)]
    out = mmr_select(chunks, top_k=10)
    assert len(out) == 3


def test_pure_relevance_when_lambda_one():
    chunks = [
        _chunk(1, doc_id=1, relevance=0.9),
        _chunk(2, doc_id=1, relevance=0.85),
        _chunk(3, doc_id=2, relevance=0.5),
    ]
    out = mmr_select(chunks, top_k=2, lambda_=1.0)
    # No diversity penalty → top two by relevance, both from doc 1.
    assert [c.chunk_id for c in out] == [1, 2]


def test_diversity_picks_new_doc_over_higher_relevance_same_doc():
    chunks = [
        _chunk(1, doc_id=1, relevance=0.9),
        _chunk(2, doc_id=1, relevance=0.85),
        _chunk(3, doc_id=2, relevance=0.5),
    ]
    out = mmr_select(chunks, top_k=2, lambda_=0.5)
    # First pick: chunk 1 (highest relevance, no penalty yet).
    # Second pick: chunk 2 score = 0.5*0.85 - 0.5*1 = -0.075.
    # Chunk 3 score = 0.5*0.5 - 0.5*0 = 0.25 → wins.
    assert [c.chunk_id for c in out] == [1, 3]


def test_pure_diversity_when_lambda_zero():
    chunks = [
        _chunk(1, doc_id=1, relevance=0.99),
        _chunk(2, doc_id=2, relevance=0.10),
        _chunk(3, doc_id=3, relevance=0.05),
    ]
    out = mmr_select(chunks, top_k=3, lambda_=0.0)
    # First pick: any (relevance term is zero) — implementation picks first highest, chunk 1.
    # Subsequent picks dominated by penalty → new docs always win.
    assert {c.document_id for c in out} == {1, 2, 3}
