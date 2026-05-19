from __future__ import annotations

from app.config import settings
from app.services.embeddings.base import EmbeddingError, EmbeddingProvider
from app.services.embeddings.openai_provider import OpenAIEmbeddingProvider

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "embedding_provider_for",
    "get_embedder",
    "reset_embedding_provider_for_tests",
]


_default_provider: EmbeddingProvider | None = None


def embedding_provider_for(name: str | None = None) -> EmbeddingProvider:
    """Return a singleton embedding provider by name.

    Defaults to `settings.embedding_provider` (env var `EMBEDDING_PROVIDER`).
    Singleton avoids re-instantiating HTTP clients per ingestion batch.
    """
    global _default_provider
    chosen = name or settings.embedding_provider

    if (
        _default_provider is not None
        and getattr(_default_provider, "_provider_name", None) == chosen
    ):
        return _default_provider

    if chosen == "openai":
        provider = OpenAIEmbeddingProvider()
    else:
        raise ValueError(f"Unknown embedding provider: {chosen!r}. Supported: 'openai'.")

    # Tag for singleton identity check above.
    provider._provider_name = chosen  # type: ignore[attr-defined]
    _default_provider = provider
    return provider


def get_embedder() -> EmbeddingProvider:
    """Convenience alias for the default embedding provider."""
    return embedding_provider_for()


def reset_embedding_provider_for_tests() -> None:
    """Test hook: clear singleton between cases."""
    global _default_provider
    _default_provider = None
