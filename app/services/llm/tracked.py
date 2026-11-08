"""The provider `llm_service_for` hands out: the real one, recording the outcome
of each call under its role in `model_status` (BE-14)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.services.llm.base import LLMProvider, LLMResponse, LLMStreamDelta, T
from app.services.model_status import Role, model_status


class TrackedProvider:
    def __init__(self, inner: LLMProvider, role: Role) -> None:
        self.inner = inner
        self.role = role

    async def generate(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        try:
            response = await self.inner.generate(
                system=system,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            model_status.failed(self.role, model, exc)
            raise
        model_status.ok(self.role, model)
        return response

    async def generate_structured(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        response_model: type[T],
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> tuple[T, LLMResponse]:
        try:
            result = await self.inner.generate_structured(
                system=system,
                messages=messages,
                model=model,
                response_model=response_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            model_status.failed(self.role, model, exc)
            raise
        model_status.ok(self.role, model)
        return result

    async def generate_stream(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> AsyncIterator[LLMStreamDelta]:
        # Recorded once: on the final delta, or at the end of a stream without
        # one. A reader that stops early (a client gone) records nothing:
        # closing the stream raises GeneratorExit, not an Exception.
        recorded = False
        try:
            async for delta in self.inner.generate_stream(
                system=system,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                if delta.final is not None and not recorded:
                    recorded = True
                    model_status.ok(self.role, model)
                yield delta
        except Exception as exc:
            if not recorded:
                model_status.failed(self.role, model, exc)
            raise
        if not recorded:
            model_status.ok(self.role, model)
