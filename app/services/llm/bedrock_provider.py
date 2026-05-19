import json
import re
import time
from collections.abc import AsyncIterator
from typing import TypeVar

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.services.llm.base import LLMResponse, LLMStreamDelta

T = TypeVar("T", bound=BaseModel)


_BEDROCK_RETRY_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
    "ModelTimeoutException",
}


def _is_bedrock_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in _BEDROCK_RETRY_CODES
    return isinstance(exc, TimeoutError)


_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, min=0.3, max=5),
    retry=retry_if_exception(_is_bedrock_retryable),
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


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    # ChatBedrockConverse returns list of {"type":"text","text":...} blocks
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


class BedrockProvider:
    """AWS Bedrock via langchain-aws ChatBedrockConverse (for Claude)."""

    def __init__(self) -> None:
        boto_cfg = BotoConfig(
            read_timeout=settings.bedrock_timeout_seconds,
            connect_timeout=10,
            retries={"max_attempts": 1, "mode": "standard"},
        )
        kwargs = {"region_name": settings.aws_region, "config": boto_cfg}
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        self._client = boto3.client("bedrock-runtime", **kwargs)

    def _build_chat(self, model: str, temperature: float, max_tokens: int) -> ChatBedrockConverse:
        return ChatBedrockConverse(
            client=self._client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @_retry
    async def _invoke_raw(self, chat: ChatBedrockConverse, lc_messages: list):
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
            text=_extract_text(response.content),
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
        chat = self._build_chat(model, temperature, max_tokens)
        lc_messages = _to_lc_messages(system, messages)
        t0 = time.perf_counter()
        full_text = ""
        usage_in = 0
        usage_out = 0
        async for chunk in chat.astream(lc_messages):
            delta = _extract_text(getattr(chunk, "content", "") or "")
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
            raw_text = _extract_text(getattr(raw, "content", ""))
            parsed = _salvage_structured(raw_text, response_model)
            if parsed is None:
                raise ValueError(
                    f"bedrock_structured_output_parse_failed: "
                    f"parsing_error={parsing_error!r} raw={raw_text!r}"
                )
        usage = getattr(raw, "usage_metadata", None) or {}
        return parsed, LLMResponse(
            text=_extract_text(raw.content),
            tokens_input=usage.get("input_tokens", 0),
            tokens_output=usage.get("output_tokens", 0),
            model=model,
            latency_ms=latency,
        )


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


def _salvage_structured(text: str, response_model: type[T]) -> T | None:
    """Best-effort recovery when langchain's `with_structured_output` returned
    `parsed=None` but the raw text still carries JSON.

    Mirrors the Gemini provider's salvage helper. See `gemini_provider.py`.
    """
    if not text:
        return None

    candidates: list[str] = []

    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())

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
