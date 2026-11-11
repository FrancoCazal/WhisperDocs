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
import math
from dataclasses import replace
from typing import Protocol

from app.logging_config import get_logger
from app.services.retriever import RetrievedChunk

logger = get_logger(__name__)


def _sigmoid(logit: float) -> float:
    # Split so a large negative logit does not overflow exp.
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    z = math.exp(logit)
    return z / (1.0 + z)


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
    """Cross-encoder via sentence-transformers, inference in a worker thread.

    A chunk's `relevance` is the sigmoid of the model's logit for (query,
    chunk): a probability that holds across queries (audit R16). It used to
    be min-max normalized over the batch, which put the best chunk at 1.0 and
    the worst at 0.0 even when none of them answered the question. The model
    is asked for its raw logit, so the score does not depend on the activation
    a model's config sets, and the sigmoid is applied here.

    `score_threshold` drops the chunks below it, but never the best one: a
    threshold that could leave a turn with no chunks would be the relevance
    gate, which is out of V1's scope. The model loads at startup when the
    reranker is on (`load`, app/main.py), not on the first question."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        score_threshold: float = 0.0,
    ):
        self.model_name = model_name
        self.score_threshold = score_threshold
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
            from torch import nn
        except ImportError as exc:
            raise RuntimeError(
                "LocalCrossEncoderReranker requires the 'rerank' install extra "
                "(pip install -e .[rerank]) for sentence-transformers."
            ) from exc
        logger.info("reranker_loading_model", model=self.model_name)
        self._model = CrossEncoder(self.model_name, activation_fn=nn.Identity())
        return self._model

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = self.load()
        pairs = [(query, c.text) for c in chunks]

        logits = await asyncio.to_thread(model.predict, pairs)

        scored: list[tuple[float, RetrievedChunk]] = [
            (_sigmoid(float(logit)), chunk) for logit, chunk in zip(logits, chunks, strict=True)
        ]
        scored.sort(key=lambda kv: kv[0], reverse=True)

        result: list[RetrievedChunk] = []
        dropped = 0
        for rank, (score, chunk) in enumerate(scored[:top_k]):
            if rank > 0 and score < self.score_threshold:
                dropped += 1
                continue
            # `replace` keeps the chunk's cosine; only the ordering score changes.
            result.append(replace(chunk, relevance=round(score, 4)))
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
