"""Backward-compatibility shim. Real implementation lives in
`app/services/embeddings/`. New code should import the provider abstraction
directly:

    from app.services.embeddings import embedding_provider_for, get_embedder

The `EmbeddingService` alias is kept so existing callers in agent nodes and
ingestion keep working without changes.
"""

from __future__ import annotations

from app.services.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    embedding_provider_for,
    get_embedder,
)

EmbeddingService = OpenAIEmbeddingProvider

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingService",
    "embedding_provider_for",
    "get_embedder",
]
