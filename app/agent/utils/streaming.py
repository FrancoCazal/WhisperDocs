"""Token-streaming helper for generation nodes.

Wraps `LLMProvider.generate_stream`, dispatches each token delta as a LangGraph
custom event ("agent_token") so the SSE endpoint can forward them to the
client. Returns the final aggregated `LLMResponse` so callers retain the same
shape they had with `generate()`.

When `settings.agent.streaming_tokens` is False, falls back to non-streaming
`generate()` — same return shape, no events dispatched.
"""

from __future__ import annotations

from langchain_core.callbacks import adispatch_custom_event

from app.config import settings
from app.logging_config import get_logger
from app.services.llm.base import LLMProvider, LLMResponse

logger = get_logger(__name__)

TOKEN_EVENT_NAME = "agent_token"


async def stream_generate(
    provider: LLMProvider,
    *,
    system: str,
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    node: str,
) -> LLMResponse:
    """Generate via streaming when enabled; dispatch token deltas as custom events.

    `node` is included in each event so the frontend can attribute tokens to
    the right turn (e.g. answer_direct vs search_docs).
    """
    if not settings.agent.streaming_tokens:
        return await provider.generate(
            system=system,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    accumulated = ""
    final: LLMResponse | None = None
    dispatch_supported = True
    try:
        async for delta in provider.generate_stream(
            system=system,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if delta.text_delta:
                accumulated += delta.text_delta
                if dispatch_supported:
                    try:
                        await adispatch_custom_event(
                            TOKEN_EVENT_NAME,
                            {"node": node, "delta": delta.text_delta},
                        )
                    except RuntimeError:
                        # No LangGraph parent run (direct node invocation, tests).
                        dispatch_supported = False
            if delta.final is not None:
                final = delta.final
    except Exception:
        logger.exception("stream_generate_failed", node=node, model=model)
        raise

    if final is not None:
        return final
    return LLMResponse(
        text=accumulated,
        tokens_input=0,
        tokens_output=0,
        model=model,
        latency_ms=0,
    )
