import logging
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.prompts import SUMMARIZE_SYSTEM_PROMPT, SUMMARY_SECTIONS
from app.agent.state import AgentState
from app.agent.utils.messages import SCOPE, format_messages_with_sources
from app.agent.utils.turn_errors import errors_so_far, turn_error
from app.config import settings
from app.db import SessionLocal
from app.models import Collection, Document
from app.services.llm import llm_service_for

logger = logging.getLogger(__name__)

_SPEAKERS = {"user": "User", "assistant": "Assistant"}

# Room for the 500 tokens the prompt asks for; a summary cut short misses its
# last section and is not kept.
SUMMARY_MAX_TOKENS = 800


async def _scope_names(
    messages: Sequence[BaseMessage],
    sessions: Callable[[], AbstractAsyncContextManager[AsyncSession]] = SessionLocal,
) -> dict[tuple[str, int], str]:
    """The names of the collections and documents the user messages searched,
    keyed ("collection", id) and ("document", id). Empty when the lookup
    fails: the transcript then names them by id."""
    ids: dict[str, set[int]] = {"collection": set(), "document": set()}
    for m in messages:
        scope = m.additional_kwargs.get(SCOPE) if isinstance(m, HumanMessage) else None
        for kind in ids:
            value = (scope or {}).get(f"{kind}_id")
            if isinstance(value, int):
                ids[kind].add(value)
    if not ids["collection"] and not ids["document"]:
        return {}
    names: dict[tuple[str, int], str] = {}
    try:
        async with sessions() as db:
            if ids["collection"]:
                rows = await db.execute(
                    select(Collection.id, Collection.name).where(
                        Collection.id.in_(ids["collection"])
                    )
                )
                names.update({("collection", i): name for i, name in rows.all()})
            if ids["document"]:
                rows = await db.execute(
                    select(Document.id, Document.title).where(Document.id.in_(ids["document"]))
                )
                names.update({("document", i): title for i, title in rows.all()})
    except Exception:
        logger.warning("summarize_scope_names_failed", exc_info=True)
        return {}
    return names


def _scope_label(scope: Any, names: dict[tuple[str, int], str]) -> str | None:
    """Where a user message's turn searched: a document (which implies its
    collection), a collection, or all of them. None for a message from before
    messages carried it."""
    if not isinstance(scope, dict):
        return None
    for kind in ("document", "collection"):
        value = scope.get(f"{kind}_id")
        if isinstance(value, int):
            name = names.get((kind, value))
            return f'{kind} "{name}"' if name else f"{kind} {value}"
    return "all collections"


def _render_transcript(messages: Sequence[BaseMessage], names: dict[tuple[str, int], str]) -> str:
    """Render messages as a plain-text transcript for the summarizer. Each
    user line names, in brackets, what its turn searched."""
    lines = []
    for m, formatted in format_messages_with_sources(messages):
        role = formatted["role"]
        if role not in _SPEAKERS:
            continue
        label = _scope_label(m.additional_kwargs.get(SCOPE), names) if role == "user" else None
        speaker = f"{_SPEAKERS[role]} [{label}]" if label else _SPEAKERS[role]
        lines.append(f"{speaker}: {formatted['content']}")
    return "\n\n".join(lines)


def _is_complete(summary: str) -> bool:
    """Every section, in order: a summary cut at `max_tokens` misses the
    last one, and one that ignored the format misses them all."""
    position = 0
    for heading in SUMMARY_SECTIONS:
        position = summary.find(heading, position)
        if position < 0:
            return False
        position += len(heading)
    return True


async def summarize_node(state: AgentState) -> dict:
    """Progressive summary that PRESERVES literal transcript (#25).

    Old behavior dropped messages[:-k] via RemoveMessage and re-injected the
    summary as a pinned SystemMessage. That broke replay/eval — the literal
    older turns were lost.

    New behavior:
      - Compute the uncovered window: messages[summary_covered_count:].
      - Absorb the uncovered prefix (everything except the trailing K) into a
        new progressive summary that also incorporates the prior summary.
      - Update `summary` and `summary_covered_count`. Never delete messages.
        Generation/classifier nodes consume `summary + messages[covered:]`
        via `build_prompt_messages`.

    The window goes to the model as text inside a single user message, not as
    replayed turns. The window ends right before a user message, so it ends
    with the assistant's; replayed as turns, Claude reads that last message as
    a prefill to continue and returns nothing.

    Each user line says which collection or document its turn searched, and
    the summary keeps that source on the facts (audit A17): after a switch,
    facts from the earlier collection read as such.

    On failure (the call raised, or the summary came back empty or missing a
    section): log, record it in `turn_errors`, keep the previous summary, and
    set `summary_retry_at` so `check_compaction` waits
    `compaction_retry_messages` messages before trying again, instead of
    paying for a failing call on every turn (audit A17). The user's turn
    proceeds either way.
    """
    messages = state["messages"]
    k = settings.agent.k_recent
    covered = state.get("summary_covered_count") or 0

    # Want to leave the trailing K messages uncompacted, and the uncompacted
    # tail must start with a user message: Bedrock Converse rejects a
    # conversation whose first turn is the assistant's. Compaction runs when the
    # last message is the user's, so an odd count is the common case and
    # `len - k` would land on an AIMessage for an even k.
    new_covered = max(len(messages) - k, 0)
    while new_covered < len(messages) and not isinstance(messages[new_covered], HumanMessage):
        new_covered += 1
    if new_covered <= covered:
        return {"pending_action": None}

    to_compact = messages[covered:new_covered]
    if not to_compact:
        return {"pending_action": None}

    prior_summary = state.get("summary") or ""
    prior_section = (
        f"Previous summary (absorb into the new one):\n{prior_summary}\n\n" if prior_summary else ""
    )

    transcript = _render_transcript(to_compact, await _scope_names(to_compact))
    if not transcript:
        return {"pending_action": None}
    transcript_messages = [
        {
            "role": "user",
            "content": (
                f"{prior_section}Conversation to summarize:\n\n"
                f"<transcript>\n{transcript}\n</transcript>\n\n"
                "Summarize the conversation so far using the required structure."
            ),
        }
    ]

    provider, model, temperature = llm_service_for("summarize")

    def failed(exc: BaseException) -> dict:
        prev = state.get("metadata") or {}
        return {
            "pending_action": None,
            "summary_retry_at": len(messages) + settings.agent.compaction_retry_messages,
            "metadata": {
                **prev,
                "turn_errors": [*errors_so_far(state), turn_error("summarize", exc, model)],
            },
        }

    try:
        resp = await provider.generate(
            system=SUMMARIZE_SYSTEM_PROMPT,
            messages=transcript_messages,
            model=model,
            temperature=temperature,
            max_tokens=SUMMARY_MAX_TOKENS,
        )
    except Exception as exc:
        logger.exception("summarize_failed, leaving history intact")
        return failed(exc)

    new_summary = (resp.text or "").strip()
    if not _is_complete(new_summary):
        logger.warning(
            "summarize_incomplete, leaving history intact",
            extra={"chars": len(new_summary), "tokens_output": resp.tokens_output},
        )
        return failed(OutputParserException("summary missing a section"))

    return {
        "summary": new_summary,
        "summary_covered_count": new_covered,
        "summary_retry_at": None,
        "pending_action": None,
    }
