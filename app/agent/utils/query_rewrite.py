"""Query rewriting for retrieval (§3.6).

Takes conversation history + latest user message and produces a dense query
optimized for vector retrieval. Resolves elliptical follow-ups.
"""

from collections.abc import Sequence
from typing import NamedTuple

from langchain_core.messages import BaseMessage

from app.agent.prompts import QUERY_REWRITE_SYSTEM_PROMPT
from app.agent.utils.messages import drop_leading_assistant, format_messages
from app.agent.utils.turn_errors import TurnError, turn_error
from app.logging_config import get_logger
from app.services.llm import llm_service_for

logger = get_logger(__name__)


class RewrittenQuery(NamedTuple):
    query: str
    # Set when the model call failed and `query` is the message as written.
    error: TurnError | None = None


async def rewrite_query_for_retrieval(
    history: Sequence[BaseMessage],
    current: str,
    *,
    summary: str | None = None,
) -> RewrittenQuery:
    """Return a dense search query. Falls back to `current` on any error,
    which it returns as `error` for the turn's `turn_errors`.

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
        return RewrittenQuery(current)

    model: str | None = None
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
    except Exception as exc:
        error = turn_error("query_rewrite", exc, model)
        logger.exception("query_rewrite_failed", kind=error["kind"], fallback="raw_user_message")
        return RewrittenQuery(current, error)
    rewritten = (resp.text or "").strip().strip('"').strip("'")
    return RewrittenQuery(rewritten or current)
