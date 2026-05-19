"""Query rewriting for retrieval (§3.6).

Takes conversation history + latest user message and produces a dense query
optimized for vector retrieval. Resolves elliptical follow-ups.
"""

import re

from langchain_core.messages import BaseMessage

from app.agent.prompts import QUERY_REWRITE_SYSTEM_PROMPT
from app.agent.utils.messages import format_messages
from app.logging_config import get_logger
from app.services.llm import llm_service_for

logger = get_logger(__name__)


# Pronouns / coreference markers across en/es. If query has any, rewrite is needed.
_COREF_MARKERS = re.compile(
    r"\b("
    r"it|its|this|that|these|those|they|them|their|he|she|him|her|his|hers|"
    r"eso|esto|esa|ese|esas|esos|aquello|aquella|aquel|"
    r"él|ella|ellos|ellas|le|les|lo|la|los|las|su|sus|suyo|suya|"
    r"y|también|además|entonces"
    r")\b",
    re.IGNORECASE,
)


def _is_self_contained(query: str) -> bool:
    """Heuristic: short, no pronouns, no coreference markers → skip rewrite."""
    word_count = len(query.split())
    if word_count > 12:
        return False
    if _COREF_MARKERS.search(query):
        return False
    return True


async def rewrite_query_for_retrieval(
    history: list[BaseMessage],
    current: str,
    *,
    summary: str | None = None,
) -> str:
    """Return a dense search query. Falls back to `current` on any error.

    Skips the LLM call when the query looks self-contained (short + no
    coreference). Saves ~300-500ms latency + ~80 tokens per turn.

    `history` should already be sliced to the uncompacted tail by the caller —
    the optional `summary` carries everything older as a single system message
    so coreference resolution can still reach pre-compaction topics.
    """
    if not summary and len(history) <= 1 and _is_self_contained(current):
        logger.debug("query_rewrite_skipped_self_contained", query_len=len(current))
        return current

    try:
        provider, model, temperature = llm_service_for("query_rewrite")
        # Prior turns as context for coreference resolution; current appended
        # explicitly so the LLM always rewrites the right message.
        context: list[dict] = []
        if summary:
            context.append(
                {
                    "role": "system",
                    "content": "Prior conversation summary (background):\n" + summary,
                }
            )
        context.extend(format_messages(history[:-1]) if len(history) > 1 else [])
        context.append({"role": "user", "content": current})
        resp = await provider.generate(
            system=QUERY_REWRITE_SYSTEM_PROMPT,
            messages=context,
            model=model,
            temperature=temperature,
            max_tokens=80,
        )
        rewritten = (resp.text or "").strip().strip('"').strip("'")
        if not rewritten:
            return current
        return rewritten
    except Exception:
        logger.exception("query_rewrite_failed", fallback="raw_user_message")
        return current
