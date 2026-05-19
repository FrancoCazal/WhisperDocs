from app.agent.state import AgentState
from app.logging_config import get_logger

logger = get_logger(__name__)


_VALID_ACTIONS = {"search_docs", "ask_clarification", "answer_direct", "out_of_scope"}


def route_compaction(state: AgentState) -> str:
    """Pre-classifier: compact or go straight to decide_action."""
    if state.get("pending_action") == "summarize":
        return "summarize"
    return "decide_action"


def route_action(state: AgentState) -> str:
    """Post-classifier: route to one of the terminal action nodes."""
    action = state.get("pending_action")
    if action in _VALID_ACTIONS:
        return action  # type: ignore[return-value]
    logger.warning(
        "route_action_invalid",
        pending_action=action,
        fallback="ask_clarification",
        hint="bug in decide_action_node",
    )
    return "ask_clarification"
