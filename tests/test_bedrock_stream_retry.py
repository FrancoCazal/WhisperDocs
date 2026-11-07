"""Bedrock streaming retries until the first token (audit A3).

botocore runs with one attempt and `_retry` wraps only the non-streaming
calls, so a throttled stream failed the turn at once. A fake chat replaces
ChatBedrockConverse; nothing reaches AWS.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from app.services.llm import bedrock_provider
from app.services.llm.bedrock_provider import BedrockProvider


def _error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "ConverseStream")


class _FakeChat:
    """Each attempt follows a script: text chunks to yield, then an optional error."""

    def __init__(self, attempts: list[tuple[list[str], Exception | None]]) -> None:
        self.attempts = attempts
        self.calls = 0

    async def astream(self, _messages):
        texts, error = self.attempts[self.calls]
        self.calls += 1
        for t in texts:
            yield SimpleNamespace(content=t, usage_metadata=None)
        if error is not None:
            raise error
        yield SimpleNamespace(
            content="", usage_metadata={"input_tokens": 7, "output_tokens": len(texts)}
        )


@pytest.fixture
def provider(monkeypatch) -> BedrockProvider:
    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(bedrock_provider.asyncio, "sleep", no_wait)
    return BedrockProvider()


async def _run(provider, monkeypatch, chat: _FakeChat):
    monkeypatch.setattr(provider, "_build_chat", lambda *_a: chat)
    deltas = []
    async for d in provider.generate_stream(system="s", messages=[], model="m"):
        deltas.append(d)
    return deltas


async def test_throttling_before_the_first_token_is_retried(provider, monkeypatch):
    chat = _FakeChat(
        [
            ([], _error("ThrottlingException")),
            ([], _error("throttlingException")),  # as a stream event reports it
            (["Hel", "lo"], None),
        ]
    )

    deltas = await _run(provider, monkeypatch, chat)

    assert chat.calls == 3
    assert "".join(d.text_delta for d in deltas) == "Hello"
    assert deltas[-1].final.text == "Hello"
    assert deltas[-1].final.tokens_input == 7


async def test_an_error_after_the_first_token_is_not_retried(provider, monkeypatch):
    chat = _FakeChat([(["Hel"], _error("ThrottlingException")), (["Hello"], None)])

    with pytest.raises(ClientError):
        await _run(provider, monkeypatch, chat)
    assert chat.calls == 1


async def test_a_non_retryable_error_is_not_retried(provider, monkeypatch):
    chat = _FakeChat([([], _error("ValidationException")), (["ok"], None)])

    with pytest.raises(ClientError):
        await _run(provider, monkeypatch, chat)
    assert chat.calls == 1


async def test_it_gives_up_after_three_attempts(provider, monkeypatch):
    chat = _FakeChat([([], _error("ServiceUnavailableException"))] * 3)

    with pytest.raises(ClientError):
        await _run(provider, monkeypatch, chat)
    assert chat.calls == 3
