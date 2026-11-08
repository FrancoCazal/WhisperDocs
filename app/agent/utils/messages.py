"""LangChain ↔ provider message format conversion."""

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# `additional_kwargs` key on an answer that reports a failure ("I had a
# problem…"). The answer stays in state, so the thread shows what the user
# saw, but no prompt, summary or query rewrite reads it (audit N4).
FAILED_TURN = "failed"


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

    Filters out messages with empty content, and failed turns (`FAILED_TURN`).
    Maps:
      HumanMessage → "user"
      AIMessage    → "assistant"
      SystemMessage→ "system"
    Other message types are ignored (tool / function messages: out of scope V1).
    """
    out: list[dict] = []
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
        out.append({"role": role, "content": content})
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


def build_prompt_messages(state: Mapping[str, Any], *, exclude_last: bool = False) -> list[dict]:
    """Build the message list a generation/classifier node should send to its LLM.

    Compaction no longer mutates `messages` — old turns stay in state and the
    `summary_covered_count` index marks where the progressive summary takes
    over. This helper rebuilds the prompt by:
      1. Prepending `state["summary"]` as a single system message (if set).
      2. Formatting messages[summary_covered_count:] (the uncompacted tail).

    `exclude_last=True` drops the most recent message from the tail (useful for
    RAG where the user's latest turn is replaced by a context-augmented prompt).
    """
    messages: list[BaseMessage] = state.get("messages") or []
    covered = state.get("summary_covered_count") or 0
    if covered < 0:
        covered = 0
    if covered > len(messages):
        covered = len(messages)
    tail = messages[covered:]
    if exclude_last and tail:
        tail = tail[:-1]

    formatted = drop_leading_assistant(format_messages(tail))
    summary = state.get("summary")
    if summary:
        formatted = [
            {
                "role": "system",
                "content": (
                    "Prior conversation summary (background context, do not "
                    "repeat verbatim):\n" + summary
                ),
            }
        ] + formatted
    return formatted
