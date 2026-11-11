"""EMBEDDING_DIMENSION against the size of `chunks.embedding` (audit R17): a
size the column cannot hold stops the API at startup, not at the first
upload's insert."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from openai import omit
from pydantic import ValidationError

from app.config import EMBEDDING_COLUMN_DIMENSION, Settings
from app.models import Chunk
from app.services.embeddings.base import EmbeddingError
from app.services.embeddings.openai_provider import OpenAIEmbeddingProvider


def test_the_default_fits_the_column():
    s = Settings(_env_file=None, database_url="postgresql://unused")

    assert s.embedding_dimension == EMBEDDING_COLUMN_DIMENSION
    assert Chunk.__table__.c.embedding.type.dim == EMBEDDING_COLUMN_DIMENSION


def test_a_size_the_column_cannot_hold_fails_at_startup():
    with pytest.raises(ValidationError, match="chunks.embedding holds 1536-dimension vectors"):
        Settings(_env_file=None, database_url="postgresql://unused", embedding_dimension=3072)


# --- what the embedder asks for and accepts (audit R19) ---------------------------


class _Embeddings:
    """Answers with vectors of `size` floats, and keeps what each call sent."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.sent: list[dict] = []

    async def create(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0] * self.size) for _ in kwargs["input"]],
            usage=None,
        )


def _provider(model: str, size: int, dimension: int = 4) -> OpenAIEmbeddingProvider:
    provider = OpenAIEmbeddingProvider(
        api_key="test", model=model, dimension=dimension, normalize=True
    )
    provider.client = SimpleNamespace(embeddings=_Embeddings(size))  # type: ignore[assignment]
    return provider


async def test_the_size_is_asked_for_so_a_larger_model_fits_the_column():
    """text-embedding-3-large gives 3072 floats unless told otherwise, which
    `chunks.embedding` cannot hold."""
    provider = _provider("text-embedding-3-large", size=4)

    assert await provider.embed_batch(["a"]) == [[0.5, 0.5, 0.5, 0.5]]
    assert provider.client.embeddings.sent[0]["dimensions"] == 4  # type: ignore[attr-defined]


async def test_ada_is_not_sent_a_size_it_rejects():
    provider = _provider("text-embedding-ada-002", size=4)

    await provider.embed_batch(["a"])

    assert provider.client.embeddings.sent[0]["dimensions"] is omit  # type: ignore[attr-defined]


async def test_a_vector_of_another_size_is_refused_before_the_insert():
    provider = _provider("text-embedding-ada-002", size=1536, dimension=4)

    with pytest.raises(EmbeddingError, match="returned 1536-dimension vectors, not the 4"):
        await provider.embed_batch(["a"])
