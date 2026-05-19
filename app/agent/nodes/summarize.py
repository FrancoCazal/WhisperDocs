import logging

from app.agent.prompts import SUMMARIZE_SYSTEM_PROMPT
from app.agent.state import AgentState
from app.agent.utils.messages import format_messages
from app.config import settings
from app.services.llm import llm_service_for

logger = logging.getLogger(__name__)


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

    On LLM failure: log and return without compacting. The user's turn still
    proceeds; the next turn will retry compaction.
    """
    messages = state["messages"]
    k = settings.agent.k_recent
    covered = state.get("summary_covered_count") or 0

    # Want to leave the trailing K messages uncompacted.
    new_covered = max(len(messages) - k, 0)
    if new_covered <= covered:
        return {"pending_action": None}

    to_compact = messages[covered:new_covered]
    if not to_compact:
        return {"pending_action": None}

    prior_summary = state.get("summary") or ""
    prior_section = (
        f"Previous summary (absorb into the new one):\n{prior_summary}\n\n" if prior_summary else ""
    )

    transcript_messages = format_messages(to_compact)
    transcript_messages.insert(
        0,
        {
            "role": "user",
            "content": prior_section
            + "Summarize the conversation so far using the required structure.",
        },
    )

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
