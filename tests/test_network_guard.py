"""The guard in tests/conftest.py refuses providers' hosts, not the local ones."""

from __future__ import annotations

import socket

import pytest

from app.config import settings
from app.services.llm.bedrock_provider import BedrockProvider


def test_refuses_a_provider_host(network_guard):
    with pytest.raises(socket.gaierror, match="tests may not reach"):
        socket.getaddrinfo("api.openai.com", 443)
    assert network_guard == ["api.openai.com"]
    network_guard.clear()


def test_resolves_localhost():
    assert socket.getaddrinfo("localhost", 5432)


async def test_a_bedrock_call_stops_at_the_guard(network_guard, monkeypatch):
    """The real provider, unstubbed, as in the summarize test that reached it.
    The credentials are fake: should the guard fail, AWS refuses the call."""
    monkeypatch.setattr(settings, "aws_access_key_id", "test")
    monkeypatch.setattr(settings, "aws_secret_access_key", "test")
    monkeypatch.setattr(settings, "aws_region", "us-east-1")
    for var in ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_BEDROCK_RUNTIME"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(Exception):  # noqa: B017 - botocore wraps the refusal
        await BedrockProvider().generate(
            system="s",
            messages=[{"role": "user", "content": "hi"}],
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            max_tokens=5,
        )
    assert network_guard and all("bedrock-runtime" in host for host in network_guard)
    network_guard.clear()
