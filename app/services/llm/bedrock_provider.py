import asyncio
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
from app.logging_config import get_logger
from app.services.llm.base import LLMResponse, LLMStreamDelta, extract_text
from app.services.llm.model_policy import (
    THINKING_HEADROOM_TOKENS,
    accepts_custom_sampling,
    is_claude_4_7_or_later,
    reasoning_level,
)

T = TypeVar("T", bound=BaseModel)

logger = get_logger(__name__)


# Compared lowercased: ConverseStream reports errors that arrive as stream
# events with the event name as the code, e.g. "throttlingException".
_BEDROCK_RETRY_CODES = {
    "throttlingexception",
    "serviceunavailableexception",
    "internalserverexception",
    "modeltimeoutexception",
    "modelstreamerrorexception",
}

_MAX_ATTEMPTS = 3


def _is_bedrock_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code.lower() in _BEDROCK_RETRY_CODES
    return isinstance(exc, TimeoutError)


def _backoff_s(attempt: int) -> float:
    """Wait before retry number `attempt` (1-based), as `_retry` waits."""
    return min(5.0, 0.3 * 2 ** (attempt - 1))


_retry = retry(
    stop=stop_after_attempt(_MAX_ATTEMPTS),
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


_extract_text = extract_text


class BedrockProvider:
    """AWS Bedrock via langchain-aws ChatBedrockConverse (for Claude)."""

    # `gen_ai.provider.name` on its spans (app/telemetry.py).
    otel_provider_name = "aws.bedrock"

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
        kwargs: dict = {"client": self._client, "model": model, "max_tokens": max_tokens}
        if accepts_custom_sampling(model):
            kwargs["temperature"] = temperature
        if is_claude_4_7_or_later(model):
            # Adaptive thinking counts toward max_tokens (model_policy.py).
            kwargs["max_tokens"] = max_tokens + THINKING_HEADROOM_TOKENS
            effort = reasoning_level(model)
            if effort:
                # langchain-aws sends it as `thinking: adaptive` plus
                # `output_config.effort` in additionalModelRequestFields.
                kwargs["reasoning_effort"] = effort
        return ChatBedrockConverse(**kwargs)

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
        """Stream the answer, retrying like `generate` until the first token.

        botocore is set to one attempt, and `_retry` cannot wrap a generator,
        so a throttled stream used to fail the turn at once (audit A3). A
        retryable error before any text is out is retried with the same
        backoff as `_retry`. After the first token the client has already
        shown text, and a retry would start the answer over, so the error
        propagates.
        """
        chat = self._build_chat(model, temperature, max_tokens)
        lc_messages = _to_lc_messages(system, messages)
        t0 = time.perf_counter()
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            full_text = ""
            usage_in = 0
            usage_out = 0
            try:
                async for chunk in chat.astream(lc_messages):
                    delta = _extract_text(getattr(chunk, "content", "") or "")
                    if delta:
                        full_text += delta
                        yield LLMStreamDelta(text_delta=delta, final=None)
                    usage = getattr(chunk, "usage_metadata", None) or {}
                    if usage:
                        usage_in = usage.get("input_tokens", usage_in) or usage_in
                        usage_out = usage.get("output_tokens", usage_out) or usage_out
                break
            except Exception as exc:
                if full_text or attempt == _MAX_ATTEMPTS or not _is_bedrock_retryable(exc):
                    raise
                logger.warning(
                    "bedrock_stream_retry",
                    model=model,
                    attempt=attempt,
                    error=type(exc).__name__,
                )
                await asyncio.sleep(_backoff_s(attempt))
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
        if is_claude_4_7_or_later(model):
            # LangChain's default forces a tool call, and Claude Sonnet 5.5
            # rejects a forced tool choice: ask for a JSON schema instead.
            structured = chat.with_structured_output(
                _output_schema(response_model), method="json_schema", include_raw=True
            )
        else:
            structured = chat.with_structured_output(response_model, include_raw=True)
        lc_messages = _to_lc_messages(system, messages)
        t0 = time.perf_counter()
        result = await self._invoke_structured(structured, lc_messages)
        latency = int((time.perf_counter() - t0) * 1000)
        raw = result["raw"]
        parsed: T | None = None
        parsing_error = result.get("parsing_error")
        candidate = result.get("parsed")
        if isinstance(candidate, response_model):
            parsed = candidate
        elif isinstance(candidate, dict):
            # The JSON schema path returns a dict; Pydantic checks the bounds
            # the schema could not carry.
            try:
                parsed = response_model.model_validate(candidate)
            except ValidationError as exc:
                parsing_error = exc
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


# Claude's structured output does not support numeric and length constraints.
# As the anthropic SDK does, each one moves into the description, and
# Pydantic checks it on the parsed answer.
_UNSUPPORTED_CONSTRAINTS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "maxItems",
)
# Keys whose values are data, not schemas.
_DATA_KEYS = frozenset({"default", "const", "enum", "examples"})
# Keys whose values map names to schemas.
_NAMED_SCHEMAS = frozenset({"properties", "$defs", "definitions"})


def _strip_constraints(node: object) -> None:
    if isinstance(node, list):
        for item in node:
            _strip_constraints(item)
        return
    if not isinstance(node, dict):
        return
    notes = [f"{key}: {node.pop(key)}" for key in _UNSUPPORTED_CONSTRAINTS if key in node]
    if isinstance(node.get("minItems"), int) and node["minItems"] > 1:
        notes.append(f"minItems: {node.pop('minItems')}")
    if notes:
        description = node.get("description", "")
        node["description"] = f"{description}\n\n{{{', '.join(notes)}}}".strip()
    for key, value in node.items():
        if key in _DATA_KEYS:
            continue
        if key in _NAMED_SCHEMAS and isinstance(value, dict):
            for schema in value.values():
                _strip_constraints(schema)
        else:
            _strip_constraints(value)


def _output_schema(response_model: type[BaseModel]) -> dict:
    """`response_model`'s JSON schema, in the subset Claude's structured
    output accepts."""
    schema = response_model.model_json_schema()
    _strip_constraints(schema)
    return schema


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
