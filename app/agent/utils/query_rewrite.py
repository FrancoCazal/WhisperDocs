"""Query rewriting for retrieval (§3.6).

Takes conversation history + latest user message and produces a dense query
optimized for vector retrieval. Resolves elliptical follow-ups.
"""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from app.agent.prompts import QUERY_REWRITE_SYSTEM_PROMPT
from app.agent.utils.messages import drop_leading_assistant, format_messages
from app.logging_config import get_logger
from app.services.llm import llm_service_for

logger = get_logger(__name__)


async def rewrite_query_for_retrieval(
    history: Sequence[BaseMessage],
    current: str,
    *,
    summary: str | None = None,
) -> str:
    """Return a dense search query. Falls back to `current` on any error.

    Rewrites only when there is something to resolve against: earlier turns or
    a summary. A first message goes to retrieval as written, which is also how
    the eval measures retrieval. Every later message goes to the model: a
    follow-up can lean on earlier turns without a pronoun that gives it away
    ("¿se puede usar con Postgres 14?"), so no word list can tell which ones
    to skip (audit A17, N5).

    `history` should already be sliced to the uncompacted tail by the caller —
    the optional `summary` carries everything older as a single system message
    so coreference resolution can still reach pre-compaction topics.
    """
    earlier = drop_leading_assistant(format_messages(history[:-1]))
    if not summary and not earlier:
        logger.debug("query_rewrite_skipped_first_turn", query_len=len(current))
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
        context.extend(earlier)
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
