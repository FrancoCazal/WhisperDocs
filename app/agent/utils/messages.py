"""LangChain ↔ provider message format conversion."""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def format_messages(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to the [{"role","content"}] shape expected by LLMProvider.

    Filters out messages with empty content. Maps:
      HumanMessage → "user"
      AIMessage    → "assistant"
      SystemMessage→ "system"
    Other message types are ignored (tool / function messages: out of scope V1).
    """
    out: list[dict] = []
    for m in messages:
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


def build_prompt_messages(state: dict[str, Any], *, exclude_last: bool = False) -> list[dict]:
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

    formatted = format_messages(tail)
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
