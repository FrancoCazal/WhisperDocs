import logging

from langchain_core.messages import HumanMessage

from app.agent.prompts import SUMMARIZE_SYSTEM_PROMPT
from app.agent.state import AgentState
from app.agent.utils.messages import format_messages
from app.config import settings
from app.services.llm import llm_service_for

logger = logging.getLogger(__name__)

_SPEAKERS = {"user": "User", "assistant": "Assistant"}


def _render_transcript(messages: list) -> str:
    """Render messages as a plain-text transcript for the summarizer."""
    lines = [
        f"{_SPEAKERS[m['role']]}: {m['content']}"
        for m in format_messages(messages)
        if m["role"] in _SPEAKERS
    ]
    return "\n\n".join(lines)


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

    On LLM failure: log and return without compacting. The user's turn still
    proceeds; the next turn will retry compaction.
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

    transcript = _render_transcript(to_compact)
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

    try:
        resp = await provider.generate(
            system=SUMMARIZE_SYSTEM_PROMPT,
            messages=transcript_messages,
            model=model,
            temperature=temperature,
            max_tokens=400,
        )
    except Exception:
        logger.exception("summarize_failed, leaving history intact")
        return {"pending_action": None}

    new_summary = (resp.text or "").strip()
    if not new_summary:
        logger.warning("summarize produced empty output, leaving history intact")
        return {"pending_action": None}

    return {
        "summary": new_summary,
        "summary_covered_count": new_covered,
        "pending_action": None,
    }
