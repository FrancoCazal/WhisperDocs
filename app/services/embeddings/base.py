"""Embedding provider abstraction (mirrors `app/services/llm/base.py`).

Embeddings need a multi-provider story symmetric to the LLM layer: OpenAI
defaults, Bedrock Titan for AWS-resident corpora, local sentence-transformers
for offline/cost-free deployments. All providers share a Protocol so callers
(retrieval, ingestion, eval) depend on the interface, not a concrete vendor.
"""

from __future__ import annotations

from typing import Protocol


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    """Protocol for any embedding backend.

    Implementations must:
    - Return unit-norm vectors when `normalize=True` was requested at init.
    - Surface vendor-specific failures as `EmbeddingError`.
    - Expose `model_id` + `dimension` for downstream validation (retrievers
      compare query/storage embedding compatibility).
    """

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
