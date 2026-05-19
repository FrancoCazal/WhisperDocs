"""Cross-encoder reranking for retrieved chunks.

Cross-encoders score (query, passage) pairs jointly and consistently outperform
bi-encoder cosine similarity for ranking — at the cost of latency proportional
to candidate count. Use as a post-retrieval refinement step: fetch a wider
candidate pool with cheap retrieval, then rerank to top_k.

Default implementation is `NoopReranker` (identity). `LocalCrossEncoderReranker`
loads `BAAI/bge-reranker-base` via sentence-transformers — gated behind the
optional `[rerank]` install extra.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from app.logging_config import get_logger
from app.services.retriever import RetrievedChunk

logger = get_logger(__name__)


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]: ...


class NoopReranker:
    """Identity reranker: truncates to top_k and preserves input order."""

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        return chunks[:top_k]


class LocalCrossEncoderReranker:
    """Cross-encoder via sentence-transformers. Lazy model load, threadpool inference."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        score_threshold: float = 0.0,
    ):
        self.model_name = model_name
        self.score_threshold = score_threshold
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "LocalCrossEncoderReranker requires the 'rerank' install extra "
                "(pip install -e .[rerank]) for sentence-transformers."
            ) from exc
        logger.info("reranker_loading_model", model=self.model_name)
        self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = self._load()
        pairs = [(query, c.text) for c in chunks]

        scores = await asyncio.to_thread(model.predict, pairs)

        scored: list[tuple[float, RetrievedChunk]] = [
            (float(score), chunk) for score, chunk in zip(scores, chunks, strict=True)
        ]
        scored.sort(key=lambda kv: kv[0], reverse=True)

        max_score = scored[0][0] if scored else 1.0
        min_score = scored[-1][0] if scored else 0.0
        span = max_score - min_score
        result: list[RetrievedChunk] = []
        dropped = 0
        for raw_score, chunk in scored[:top_k]:
            normalized = (raw_score - min_score) / span if span > 0 else 1.0
            if normalized < self.score_threshold:
                dropped += 1
                continue
            result.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    relevance=round(float(normalized), 4),
                )
            )
        if dropped > 0:
            logger.info(
                "reranker_threshold_dropped",
                dropped=dropped,
                threshold=self.score_threshold,
                kept=len(result),
            )
        return result


_default_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """Module-level singleton for FastAPI Depends() and direct calls."""
    global _default_reranker
    if _default_reranker is not None:
        return _default_reranker

    from app.config import settings

    if settings.reranker_enabled:
        _default_reranker = LocalCrossEncoderReranker(
            model_name=settings.reranker_model,
            score_threshold=settings.reranker_score_threshold,
        )
    else:
        _default_reranker = NoopReranker()
    return _default_reranker


def reset_reranker_for_tests() -> None:
    """Test hook: clear singleton between cases."""
    global _default_reranker
    _default_reranker = NoopReranker()
