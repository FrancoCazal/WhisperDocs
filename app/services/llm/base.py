from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


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
