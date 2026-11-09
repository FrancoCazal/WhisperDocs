"""LangChain ↔ provider message format conversion."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.agent.prompts import CONVERSATION_SUMMARY
from app.agent.utils.untrusted import fill

# `additional_kwargs` key on an answer that reports a failure ("I had a
# problem…"). The answer stays in state, so the thread shows what the user
# saw, but no prompt, summary or query rewrite reads it (audit N4).
FAILED_TURN = "failed"

# `additional_kwargs` key on a user message: what its turn searched,
# {"collection_id", "document_id"}, so the summary can say where each fact
# came from (audit A17).
SCOPE = "scope"


def stamped(**additional_kwargs: Any) -> dict[str, Any]:
    """A message's `additional_kwargs` with the time it was written (UTC,
    ISO 8601), so a reopened thread shows when each turn happened."""
    return {**additional_kwargs, "timestamp": datetime.now(UTC).isoformat()}


def asked(text: str, *, collection_id: int | None, document_id: int | None = None) -> HumanMessage:
    """A user's message: stamped with its time and with what its turn
    searched (a document, a collection, or all of them when both are None)."""
    scope = {"collection_id": collection_id, "document_id": document_id}
    return HumanMessage(content=text, additional_kwargs=stamped(**{SCOPE: scope}))


def failed_answer(content: str, **additional_kwargs: Any) -> AIMessage:
    """An answer that reports a failure instead of answering."""
    return AIMessage(content=content, additional_kwargs={**additional_kwargs, FAILED_TURN: True})


def _without_failed_turns(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Drop each failed answer together with the question it failed on, and
    each question left without an answer.

    A turn cut short (the client closed the stream, the process restarted)
    leaves its question checkpointed with no answer after it. Followed by the
    next question, that is two user turns in a row, which Bedrock Converse
    rejects; dropping a failed answer alone would leave the same.
    """
    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.additional_kwargs.get(FAILED_TURN):
            if out and isinstance(out[-1], HumanMessage):
                out.pop()
            continue
        if isinstance(m, HumanMessage) and out and isinstance(out[-1], HumanMessage):
            out.pop()
        out.append(m)
    return out


def format_messages(messages: Sequence[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to the [{"role","content"}] shape expected by LLMProvider.
    See `format_messages_with_sources`."""
    return [formatted for _m, formatted in format_messages_with_sources(messages)]


def format_messages_with_sources(
    messages: Sequence[BaseMessage],
) -> list[tuple[BaseMessage, dict]]:
    """Each kept message with its [{"role","content"}] form, for a caller
    that also reads the message's `additional_kwargs`.

    Filters out messages with empty content, and failed turns (`FAILED_TURN`).
    Maps:
      HumanMessage → "user"
      AIMessage    → "assistant"
      SystemMessage→ "system"
    Other message types are ignored (tool / function messages: out of scope V1).
    """
    out: list[tuple[BaseMessage, dict]] = []
    for m in _without_failed_turns(messages):
        content = getattr(m, "content", "")
        if not isinstance(content, str):
            # Handle list-of-blocks (Bedrock-style) by joining text parts
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                content = "".join(parts)
            else:
                content = str(content)
        if not content:
            continue
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, SystemMessage):
            role = "system"
        else:
            continue
        out.append((m, {"role": role, "content": content}))
    return out


def drop_leading_assistant(formatted: list[dict]) -> list[dict]:
    """Drop assistant messages that precede the first user message.

    Providers require the conversation to open with a user turn (Bedrock
    Converse returns a ValidationException otherwise). `summarize` now aligns
    `summary_covered_count` to a user message; this guards threads whose
    checkpoint was compacted before that fix and still has an odd count.
    """
    start = 0
    while start < len(formatted) and formatted[start]["role"] == "assistant":
        start += 1
    return formatted[start:]


def with_summary(formatted: list[dict], summary: str | None) -> list[dict]:
    """The summary of earlier turns, as data at the head of the first user
    message, in a `<conversation_summary>` element it cannot close (audit
    A10). As a system message it had the instructions' authority, though it
    is written from documents and users' words. A message of its own would
    put two user turns in a row, which Bedrock Converse rejects."""
    if not summary:
        return formatted
    block = fill(CONVERSATION_SUMMARY, summary=summary)
    out = [dict(m) for m in formatted]
    for m in out:
        if m["role"] == "user":
            m["content"] = f"{block}\n\n{m['content']}"
            return out
    return [{"role": "user", "content": block}, *out]


def build_prompt_messages(state: Mapping[str, Any], *, last_user: str | None = None) -> list[dict]:
    """Build the message list a generation/classifier node should send to its LLM.

    Compaction no longer mutates `messages` — old turns stay in state and the
    `summary_covered_count` index marks where the progressive summary takes
    over. This helper rebuilds the prompt by:
      1. Formatting messages[summary_covered_count:] (the uncompacted tail).
      2. Putting `state["summary"]` (if set) at the head of its first user
         message (`with_summary`).

    `last_user` replaces the most recent message: `search_docs` sends the
    user's question inside its RAG prompt.
    """
    messages: list[BaseMessage] = state.get("messages") or []
    covered = state.get("summary_covered_count") or 0
    if covered < 0:
        covered = 0
    if covered > len(messages):
        covered = len(messages)
    tail = messages[covered:]
    if last_user is not None and tail:
        tail = tail[:-1]

    formatted = drop_leading_assistant(format_messages(tail))
    if last_user is not None:
        formatted.append({"role": "user", "content": last_user})
    return with_summary(formatted, state.get("summary"))
