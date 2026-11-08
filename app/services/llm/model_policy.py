"""Per-model request policy.

Providers ask this module about a model before building a request:

- Does it accept custom sampling parameters (`temperature`)? Claude 4.7 and
  later accept only the defaults and return a 400 for any other value.
  Gemini 3.6 Flash, Gemini 3.5 Flash-Lite and every later Gemini release use
  fixed sampling, and Google recommends the default temperature for the whole
  Gemini 3 line. Older models keep the per-node temperatures from
  `AgentSettings`.
- Is it Claude 4.7 or later? On Bedrock those models take an effort level,
  think adaptively, and count that thinking toward `max_tokens` even though
  its text is not returned. Claude Sonnet 5.5 also rejects a forced tool
  choice, which is how LangChain asks for structured output by default.
  `BedrockProvider` sends them `BEDROCK_CLAUDE_EFFORT`, adds
  `THINKING_HEADROOM_TOKENS` to `max_tokens`, and asks for structured output
  as a JSON schema (Bedrock's native structured output).
- Which effort or thinking level goes with it (`reasoning_level`). The
  providers send it, and the call's span records it.
"""

from __future__ import annotations

import re

from app.config import settings

# claude-sonnet-4-5-20250929 -> (4, 5); claude-sonnet-4-20250514 -> (4, 0);
# anthropic.claude-opus-5-5 -> (5, 5). The minor version is one or two digits
# that are not the start of an 8-digit date.
_CLAUDE_TIERED = re.compile(r"claude-(?:opus|sonnet|haiku)-(\d+)(?:-(\d{1,2})(?!\d))?")
# Families that started at version 5.
_CLAUDE_V5_FAMILIES = re.compile(r"claude-(?:fable|mythos)-")
_GEMINI_MAJOR = re.compile(r"gemini-(\d+)")

# Added to a Claude 4.7+ call's `max_tokens`. At effort `low` the model skips
# thinking on most simple requests; when it does think, the answer still gets
# the tokens the caller asked for. Only the tokens produced are billed.
THINKING_HEADROOM_TOKENS = 2048


def claude_version(model: str) -> tuple[int, int] | None:
    """Return (major, minor) for a Claude model ID, or None if it is not one."""
    normalized = model.lower()
    if _CLAUDE_V5_FAMILIES.search(normalized):
        return (5, 0)
    match = _CLAUDE_TIERED.search(normalized)
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2) or 0))


def is_claude_4_7_or_later(model: str) -> bool:
    version = claude_version(model)
    return version is not None and version >= (4, 7)


def is_gemini_3_or_later(model: str) -> bool:
    match = _GEMINI_MAJOR.search(model.lower())
    return match is not None and int(match.group(1)) >= 3


def accepts_custom_sampling(model: str) -> bool:
    """Whether a non-default `temperature` may be sent for this model."""
    return not (is_claude_4_7_or_later(model) or is_gemini_3_or_later(model))


def reasoning_level(model: str) -> str | None:
    """The effort or thinking level sent with a call to this model, if any:
    `BEDROCK_CLAUDE_EFFORT` for Claude 4.7+, `GEMINI_THINKING_LEVEL` for
    Gemini 3+. None sends nothing, and the model uses its default."""
    if is_claude_4_7_or_later(model):
        return settings.bedrock_claude_effort
    if is_gemini_3_or_later(model):
        return settings.gemini_thinking_level
    return None
