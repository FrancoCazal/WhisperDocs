from langchain_core.messages import HumanMessage

from app.agent.state import AgentState
from app.agent.token_utils import count_tokens_estimate
from app.config import settings


async def check_compaction_node(state: AgentState) -> dict:
    """Deterministic pre-classifier: (1) reset transient fields, (2) decide compaction.

    Reset is critical — the checkpointer persists the full state, so without
    explicit reset `retrieved_chunks` and `metadata` from a previous RAG turn
    would leak into a subsequent answer_direct turn.

    Compaction guards:
      - Only compact if last message is from user (don't compact mid-turn).
      - Trigger on token OR message-count threshold.
      - Not before `summary_retry_at` messages, after a failed summary.
    """
    messages = state["messages"]
    covered = state.get("summary_covered_count") or 0
    uncovered = messages[covered:]
    last_is_user = bool(uncovered) and isinstance(uncovered[-1], HumanMessage)
    total_tokens = count_tokens_estimate(uncovered)
    over_threshold = (
        total_tokens > settings.agent.compaction_token_threshold
        or len(uncovered) > settings.agent.compaction_count_threshold
    )
    retry_at = state.get("summary_retry_at")
    backing_off = retry_at is not None and len(messages) < retry_at
    needs_compaction = over_threshold and last_is_user and not backing_off

    return {
        "pending_action": "summarize" if needs_compaction else None,
        "retrieved_chunks": None,
        "metadata": {},
    }
