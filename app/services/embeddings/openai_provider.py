from __future__ import annotations

import math

from openai import AsyncOpenAI, omit

from app import telemetry
from app.config import settings
from app.services.embeddings.base import EmbeddingError
from app.services.model_status import model_status


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return vec
    return [x / norm for x in vec]


def takes_dimensions(model: str) -> bool:
    """Whether the model takes `dimensions`: the text-embedding-3 models and
    later do; text-embedding-ada-002 rejects it."""
    return not model.startswith("text-embedding-ada")


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
        self.client = AsyncOpenAI(api_key=key, timeout=settings.embedding_timeout_seconds)
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
        """The texts' vectors, `dimension` floats each (audit R19). The size is
        asked for, so text-embedding-3-large gives 1536 and not its 3072, and
        a vector of another size is refused here, not at the insert."""
        if not texts:
            return []

        sized = self._dimension if takes_dimensions(self._model) else omit
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            with telemetry.embeddings(provider="openai", model=self._model) as call:
                try:
                    response = await self.client.embeddings.create(
                        model=self._model, input=batch, dimensions=sized
                    )
                except Exception as exc:
                    model_status.failed("embedding", self._model, exc)
                    raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc
                if response.usage is not None:
                    call.usage(response.usage.prompt_tokens)

            for item in response.data:
                vec = item.embedding
                if len(vec) != self._dimension:
                    error = EmbeddingError(
                        f"{self._model} returned {len(vec)}-dimension vectors, not the "
                        f"{self._dimension} of EMBEDDING_DIMENSION"
                    )
                    model_status.failed("embedding", self._model, error)
                    raise error
                if self.normalize:
                    vec = _l2_normalize(vec)
                all_vectors.append(vec)

        model_status.ok("embedding", self._model)
        return all_vectors
