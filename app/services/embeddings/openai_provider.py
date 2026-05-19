from __future__ import annotations

import math

from openai import AsyncOpenAI

from app.config import settings
from app.services.embeddings.base import EmbeddingError


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return vec
    return [x / norm for x in vec]


class OpenAIEmbeddingProvider:
    """OpenAI-backed embedding provider with batching and L2 normalization."""

    BATCH_SIZE = 100

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        normalize: bool | None = None,
        dimension: int | None = None,
    ):
        key = api_key or settings.openai_api_key
        if not key:
            raise EmbeddingError("OPENAI_API_KEY not configured")
        self.client = AsyncOpenAI(api_key=key)
        self._model = model or settings.embedding_model
        self._dimension = dimension if dimension is not None else settings.embedding_dimension
        self.normalize = settings.embedding_l2_normalize if normalize is None else normalize

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            try:
                response = await self.client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
            except Exception as exc:
                raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc

            for item in response.data:
                vec = item.embedding
                if self.normalize:
                    vec = _l2_normalize(vec)
                all_vectors.append(vec)

        return all_vectors
