"""The provider `llm_service_for` hands out: the real one, recording the outcome
of each call under its role in `model_status` (BE-14), and each call as a
`chat {model}` span (app/telemetry.py)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractContextManager

from app import telemetry
from app.services.llm.base import LLMProvider, LLMResponse, LLMStreamDelta, T
from app.services.model_status import Role, model_status


class TrackedProvider:
    def __init__(self, inner: LLMProvider, role: Role) -> None:
        self.inner = inner
        self.role = role

    def _span(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        *,
        stream: bool = False,
        structured: bool = False,
    ) -> AbstractContextManager[telemetry.ModelCall]:
        return telemetry.chat(
            # BedrockProvider and GeminiProvider name themselves as the
            # conventions do; a test double goes by its class name.
            provider=getattr(self.inner, "otel_provider_name", type(self.inner).__name__),
            model=model,
            role=self.role,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            structured=structured,
        )

    async def generate(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        with self._span(model, temperature, max_tokens) as call:
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
            call.usage(response.tokens_input, response.tokens_output, response.model)
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
        with self._span(model, temperature, max_tokens, structured=True) as call:
            try:
                parsed, response = await self.inner.generate_structured(
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
            call.usage(response.tokens_input, response.tokens_output, response.model)
        model_status.ok(self.role, model)
        return parsed, response

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
        # closing the stream raises GeneratorExit, not an Exception. The span
        # ends either way.
        recorded = False
        with self._span(model, temperature, max_tokens, stream=True) as call:
            try:
                async for delta in self.inner.generate_stream(
                    system=system,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    if delta.text_delta:
                        call.first_chunk()
                    if delta.final is not None and not recorded:
                        recorded = True
                        final = delta.final
                        call.usage(final.tokens_input, final.tokens_output, final.model)
                        model_status.ok(self.role, model)
                    yield delta
            except Exception as exc:
                if not recorded:
                    model_status.failed(self.role, model, exc)
                raise
        if not recorded:
            model_status.ok(self.role, model)
