import json
import re
import time
from collections.abc import AsyncIterator
from typing import TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.services.llm.base import LLMResponse, LLMStreamDelta

T = TypeVar("T", bound=BaseModel)


try:
    from google.api_core import exceptions as google_exceptions  # type: ignore

    _GEMINI_RETRY_EXCS: tuple[type[BaseException], ...] = (
        TimeoutError,
        google_exceptions.ResourceExhausted,
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
    )
except ImportError:
    _GEMINI_RETRY_EXCS = (TimeoutError,)


_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, min=0.3, max=5),
    retry=retry_if_exception_type(_GEMINI_RETRY_EXCS),
    reraise=True,
)


def _to_lc_messages(system: str, messages: list[dict]) -> list:
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


class GeminiProvider:
    """Gemini via Vertex AI (ADC auth, billed against GCP project credits).

    Uses unified `ChatGoogleGenerativeAI` with `vertexai=True` — the post-3.2
    successor to `ChatVertexAI`. Same auth (ADC / service-account), same billing.
    """

    def _build_chat(
        self, model: str, temperature: float, max_tokens: int
    ) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=model,
            vertexai=True,
            project=settings.gcp_project or None,
            location=settings.gcp_location,
            temperature=temperature,
            max_output_tokens=max_tokens,
            timeout=settings.gemini_timeout_seconds,
        )

    @_retry
    async def _invoke_raw(self, chat: ChatGoogleGenerativeAI, lc_messages: list):
        return await chat.ainvoke(lc_messages)

    async def generate(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        chat = self._build_chat(model, temperature, max_tokens)
        lc_messages = _to_lc_messages(system, messages)
        t0 = time.perf_counter()
        response = await self._invoke_raw(chat, lc_messages)
        latency = int((time.perf_counter() - t0) * 1000)
        usage = getattr(response, "usage_metadata", None) or {}
        return LLMResponse(
            text=response.content if isinstance(response.content, str) else str(response.content),
            tokens_input=usage.get("input_tokens", 0),
            tokens_output=usage.get("output_tokens", 0),
            model=model,
            latency_ms=latency,
        )

    async def generate_stream(
        self,
        *,
        system: str,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> AsyncIterator[LLMStreamDelta]:
        chat: ChatGoogleGenerativeAI = self._build_chat(model, temperature, max_tokens)
        lc_messages = _to_lc_messages(system, messages)
        t0 = time.perf_counter()
        full_text = ""
        usage_in = 0
        usage_out = 0
        async for chunk in chat.astream(lc_messages):
            content = getattr(chunk, "content", "") or ""
            delta = content if isinstance(content, str) else str(content)
            if delta:
                full_text += delta
                yield LLMStreamDelta(text_delta=delta, final=None)
            usage = getattr(chunk, "usage_metadata", None) or {}
            if usage:
                usage_in = usage.get("input_tokens", usage_in) or usage_in
                usage_out = usage.get("output_tokens", usage_out) or usage_out
        latency = int((time.perf_counter() - t0) * 1000)
        yield LLMStreamDelta(
            text_delta="",
            final=LLMResponse(
                text=full_text,
                tokens_input=usage_in,
                tokens_output=usage_out,
                model=model,
                latency_ms=latency,
            ),
        )

    @_retry
    async def _invoke_structured(self, structured, lc_messages: list):
        return await structured.ainvoke(lc_messages)

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
        chat = self._build_chat(model, temperature, max_tokens)
        structured = chat.with_structured_output(response_model, include_raw=True)
        lc_messages = _to_lc_messages(system, messages)
        t0 = time.perf_counter()
        result = await self._invoke_structured(structured, lc_messages)
        latency = int((time.perf_counter() - t0) * 1000)
        raw = result["raw"]
        parsed: T | None = result.get("parsed")
        parsing_error = result.get("parsing_error")

        if parsed is None:
            raw_text = getattr(raw, "content", "")
            text_str = raw_text if isinstance(raw_text, str) else str(raw_text)
            parsed = _salvage_structured(text_str, response_model)
            if parsed is None:
                raise ValueError(
                    f"gemini_structured_output_parse_failed: "
                    f"parsing_error={parsing_error!r} raw={text_str!r}"
                )

        usage = getattr(raw, "usage_metadata", None) or {}
        return parsed, LLMResponse(
            text=raw.content if isinstance(raw.content, str) else str(raw.content),
            tokens_input=usage.get("input_tokens", 0),
            tokens_output=usage.get("output_tokens", 0),
            model=model,
            latency_ms=latency,
        )


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


def _salvage_structured(text: str, response_model: type[T]) -> T | None:
    """Best-effort recovery when Gemini's `with_structured_output` returned
    `parsed=None` but the raw text still carries JSON.

    Common shapes:
      1. Markdown-fenced: ```json\n{...}\n```
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
