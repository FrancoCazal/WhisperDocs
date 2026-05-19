"""Retrieval pipeline composing VectorRetriever + Reranker (+ MMR later).

Keeps `VectorRetriever` a pure data-access component. All optional refinement
steps (reranking, diversity) plug in here.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.services.mmr import mmr_select
from app.services.reranker import NoopReranker, Reranker, get_reranker
from app.services.retriever import RetrievedChunk, VectorRetriever


async def retrieve(
    db: AsyncSession,
    query_text: str,
    query_embedding: list[float],
    collection_id: int | None,
    top_k: int,
    reranker: Reranker | None = None,
) -> list[RetrievedChunk]:
    """Hybrid (vector + BM25) retrieval, optionally followed by cross-encoder rerank.

    When reranking is active, fetches `reranker_candidate_pool` chunks (default
    20), then narrows to `top_k` via the reranker. Without reranking, returns
    `top_k` directly from RRF fusion.
    """
    retriever = VectorRetriever(db)
    active_reranker = reranker if reranker is not None else get_reranker()
    is_noop = isinstance(active_reranker, NoopReranker)
    mmr_active = settings.mmr_enabled

    # If reranking or MMR is active, fetch a wider candidate pool so the
    # refinement step has something to work with.
    needs_pool = not is_noop or mmr_active
    fetch_k = max(top_k, settings.reranker_candidate_pool) if needs_pool else top_k
    candidates = await retriever.hybrid_retrieve(
        query_text=query_text,
        query_embedding=query_embedding,
        collection_id=collection_id,
        top_k=fetch_k,
    )

    if not candidates:
        return []

    # When reranking is active, narrow to a slightly larger window than top_k
    # so MMR has room to swap. When inactive, skip rerank entirely.
    if is_noop:
        ranked = candidates
    else:
        rerank_window = top_k * 2 if mmr_active else top_k
        ranked = await active_reranker.rerank(query_text, candidates, top_k=rerank_window)

    if mmr_active:
        return mmr_select(ranked, top_k=top_k, lambda_=settings.mmr_lambda)
    return ranked[:top_k]
