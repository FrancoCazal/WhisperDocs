import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def to_lc_messages(system: str, messages: list[dict]) -> list:
    """The system prompt and the `{"role", "content"}` messages as LangChain
    messages, for either provider."""
    lc: list = [SystemMessage(content=system)]
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "user":
            lc.append(HumanMessage(content=content))
        elif role == "assistant":
            lc.append(AIMessage(content=content))
        elif role == "system":
            lc.append(SystemMessage(content=content))
    return lc


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


def salvage_structured(text: str, response_model: type[T]) -> T | None:
    """Best-effort recovery when langchain's `with_structured_output` returned
    `parsed=None` but the raw text still carries JSON. Both providers use it.

    Common shapes:
      1. Markdown-fenced: ```json\\n{...}\\n```
      2. Prose preamble + JSON object literal
      3. Trailing commentary after JSON

    Returns None if nothing salvageable. Caller raises the original error.
    """
    if not text:
        return None

    candidates: list[str] = []

    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())

    # Greedy outermost {...} block — handles "Here is the JSON: {...}" preamble.
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first : last + 1])

    candidates.append(text.strip())

    for c in candidates:
        try:
            return response_model.model_validate_json(c)
        except (ValidationError, ValueError, json.JSONDecodeError):
            continue
    return None


def extract_text(content: object) -> str:
    """Return the text of a LangChain message `content`.

    Bedrock Converse and Gemini 3 return content as a list of blocks
    (`{"type": "text", "text": ...}`, plus reasoning or signature blocks), not a
    string. Only text blocks are kept; `str()` on the list would leak its repr.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content is not None else ""


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response shape across providers."""

    text: str
    tokens_input: int
    tokens_output: int
    model: str
    latency_ms: int


@dataclass(frozen=True)
class LLMStreamDelta:
    """One incremental update from a streaming generation.

    `text_delta` is the new token chunk (may be empty on final summary).
    `final` is set on the last delta with full token usage + latency.
    """

    text_delta: str
    final: LLMResponse | None = None


class LLMProvider(Protocol):
    async def generate(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse: ...

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        response_model: type[T],
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> tuple[T, LLMResponse]: ...

    def generate_stream(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> AsyncIterator[LLMStreamDelta]: ...
