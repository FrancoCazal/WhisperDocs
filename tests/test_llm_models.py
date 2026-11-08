"""Model configuration and per-model request policy (audit A1, A7; review N1)."""

from __future__ import annotations

import pytest

from app.config import AgentSettings, settings
from app.services.llm import validate_agent_models
from app.services.llm import bedrock_provider as bedrock_mod
from app.services.llm import gemini_provider as gemini_mod
from app.services.llm.base import extract_text
from app.services.llm.model_policy import (
    accepts_custom_sampling,
    bedrock_converse_unsupported_reason,
    claude_version,
)
from app.services.llm.pricing import estimate_cost_usd


@pytest.mark.parametrize(
    ("model", "version"),
    [
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", (4, 5)),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", (4, 5)),
        ("us.anthropic.claude-sonnet-4-20250514-v1:0", (4, 0)),
        ("anthropic.claude-opus-4-7", (4, 7)),
        ("anthropic.claude-sonnet-5-5", (5, 5)),
        ("claude-haiku-5-5", (5, 5)),
        ("anthropic.claude-fable-5-1", (5, 0)),
        ("gemini-3.8-flash", None),
    ],
)
def test_claude_version(model, version):
    assert claude_version(model) == version


@pytest.mark.parametrize(
    ("model", "accepts"),
    [
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", True),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", True),
        ("anthropic.claude-opus-4-7", False),
        ("anthropic.claude-sonnet-5-5", False),
        ("gemini-2.5-flash", True),
        ("gemini-3.5-flash-lite", False),
        ("gemini-3.8-flash", False),
    ],
)
def test_accepts_custom_sampling(model, accepts):
    assert accepts_custom_sampling(model) is accepts


def test_bedrock_converse_rejects_claude_4_7_and_later():
    assert (
        bedrock_converse_unsupported_reason("us.anthropic.claude-sonnet-4-5-20250929-v1:0") is None
    )
    reason = bedrock_converse_unsupported_reason("anthropic.claude-sonnet-5-5")
    assert reason is not None and "bedrock-mantle" in reason


def test_default_models_are_valid_and_priced(monkeypatch):
    """Every default model routes to a provider that can serve it and has a
    pricing row. Fails when a default changes without updating pricing.py."""
    # Defaults only: ignore a developer's .env overrides.
    monkeypatch.setattr(settings, "agent", AgentSettings(_env_file=None))
    errors, warnings = validate_agent_models()
    assert errors == []
    assert warnings == []


def test_validation_flags_unservable_models(monkeypatch):
    monkeypatch.setattr(settings.agent, "model_rag_generation", "anthropic.claude-sonnet-5-5")
    monkeypatch.setattr(settings.agent, "model_classifier", "gpt-5")

    errors, _warnings = validate_agent_models()

    assert any("AGENT_MODEL_RAG_GENERATION" in e and "bedrock-mantle" in e for e in errors)
    assert any("AGENT_MODEL_CLASSIFIER" in e for e in errors)


class _CaptureKwargs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_bedrock_sends_temperature_only_when_accepted(monkeypatch):
    monkeypatch.setattr(bedrock_mod, "ChatBedrockConverse", _CaptureKwargs)
    provider = bedrock_mod.BedrockProvider()

    chat = provider._build_chat("us.anthropic.claude-sonnet-4-5-20250929-v1:0", 0.3, 100)
    assert chat.kwargs["temperature"] == 0.3

    with pytest.raises(ValueError, match="bedrock-mantle"):
        provider._build_chat("anthropic.claude-sonnet-5-5", 0.3, 100)


def test_gemini_3_omits_temperature_and_lowers_thinking(monkeypatch):
    monkeypatch.setattr(gemini_mod, "ChatGoogleGenerativeAI", _CaptureKwargs)
    monkeypatch.setattr(settings, "gemini_thinking_level", "low")
    provider = gemini_mod.GeminiProvider()

    new = provider._build_chat("gemini-3.8-flash", 0.0, 100).kwargs
    assert "temperature" not in new
    assert new["thinking_level"] == "low"
    assert new["max_retries"] == 2

    old = provider._build_chat("gemini-2.5-flash", 0.0, 100).kwargs
    assert old["temperature"] == 0.0
    assert "thinking_level" not in old


@pytest.mark.parametrize(
    ("content", "text"),
    [
        ("plain", "plain"),
        ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "ab"),
        (
            [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": "answer", "extras": {"signature": "x"}},
            ],
            "answer",
        ),
        (["a", "b"], "ab"),
        (None, ""),
    ],
)
def test_extract_text(content, text):
    assert extract_text(content) == text


def test_unknown_model_uses_fallback_price():
    # $1 / $5 per 1M tokens
    assert estimate_cost_usd("some-unknown-model", 1_000_000, 1_000_000) == 6.0
    assert estimate_cost_usd(None, 10, 10) == 0.0
