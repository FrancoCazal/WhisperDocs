"""An embedding request has a time limit (BE-26). Without one, the OpenAI SDK
waits 600 s for an answer and retries twice: an endpoint that takes the
connection and never answers held a chat turn or an MCP call for half an
hour."""

from __future__ import annotations

import asyncio

import pytest

from app.config import settings
from app.services.embeddings.base import EmbeddingError
from app.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.services.llm.errors import classify_error


class _Status:
    def __init__(self) -> None:
        self.failed_with: list[BaseException] = []

    def failed(self, _kind, _model, exc) -> None:
        self.failed_with.append(exc)

    def ok(self, *_args) -> None:
        pass


async def test_an_endpoint_that_never_answers_fails_within_the_limit(monkeypatch):
    async def silent(reader, writer):
        await reader.read()  # takes the request, answers nothing
        # Closed on this side too, or the server waits for it when it closes.
        writer.close()

    server = await asyncio.start_server(silent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    status = _Status()
    monkeypatch.setattr(settings, "embedding_timeout_seconds", 1)
    monkeypatch.setattr("app.services.embeddings.openai_provider.model_status", status)
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")

    provider = OpenAIEmbeddingProvider(api_key="test-key")
    async with server:
        with pytest.raises(EmbeddingError) as failed:
            # Three tries of 1 s and the SDK's backoff between them, about
            # 5 s; without the limit, 600 s each.
            await asyncio.wait_for(provider.embed("q"), 20)

    assert provider.client.timeout == 1
    assert classify_error(failed.value.__cause__) == "timeout"
    assert len(status.failed_with) == 1
