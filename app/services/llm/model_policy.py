"""Per-model request policy.

Providers ask this module two questions before building a request:

- Does the model accept custom sampling parameters (`temperature`)?
  Claude 4.7 and later accept only the defaults and return a 400 for any other
  value. Gemini 3.6 Flash, Gemini 3.5 Flash-Lite and every later Gemini release
  use fixed sampling, and Google recommends the default temperature for the
  whole Gemini 3 line. Older models keep the per-node temperatures from
  `AgentSettings`.
- Can the Bedrock Converse API serve the model? Claude 4.7 and later run only
  on the `bedrock-mantle` Messages endpoint, which `BedrockProvider` does not
  call yet.
"""

from __future__ import annotations

import re

# claude-sonnet-4-5-20250929 -> (4, 5); claude-sonnet-4-20250514 -> (4, 0);
# anthropic.claude-opus-5-5 -> (5, 5). The minor version is one or two digits
# that are not the start of an 8-digit date.
_CLAUDE_TIERED = re.compile(r"claude-(?:opus|sonnet|haiku)-(\d+)(?:-(\d{1,2})(?!\d))?")
# Families that started at version 5.
_CLAUDE_V5_FAMILIES = re.compile(r"claude-(?:fable|mythos)-")
_GEMINI_MAJOR = re.compile(r"gemini-(\d+)")


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


def bedrock_converse_unsupported_reason(model: str) -> str | None:
    """Explain why the Converse API cannot serve this Bedrock model, if it cannot."""
    if is_claude_4_7_or_later(model):
        return (
            f"{model!r} is Claude 4.7 or later. Amazon Bedrock serves those models only "
            "through the bedrock-mantle Messages endpoint, and BedrockProvider uses the "
            "Converse API. Use a Claude 4.6-or-earlier model ID here until a "
            "bedrock-mantle provider exists."
        )
    return None
