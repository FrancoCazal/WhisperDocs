from app.agent.prompts import CLASSIFIER_SYSTEM_PROMPT
from app.agent.schemas import AgentAction
from app.agent.state import AgentState
from app.agent.utils.messages import build_prompt_messages
from app.config import settings
from app.logging_config import get_logger
from app.metrics import errors_total
from app.services.llm import llm_service_for

logger = get_logger(__name__)


async def decide_action_node(state: AgentState) -> dict:
    """LLM classifier with structured outputs → routes to one terminal node.

    Low-confidence predictions are downgraded to `classifier_low_confidence_action`
    (default `ask_clarification`) to avoid silently routing uncertain calls to
    search_docs or answer_direct.

    When the classifier's model fails, the turn goes to search_docs. It answers
    only from retrieved passages, so it cannot make things up as answer_direct
    could, and it runs on its own model. Clarification used to take these
    turns, but it calls a model too: with Vertex refusing every call (a GCP
    project without billing), it failed the same way and replied a fixed "can
    you give me more details", so every question read as vague and the outage
    did not show.
    """
    provider, model, temperature = llm_service_for("classifier")

    try:
        parsed, resp = await provider.generate_structured(
            system=CLASSIFIER_SYSTEM_PROMPT,
            messages=build_prompt_messages(state),
            model=model,
            response_model=AgentAction,
            temperature=temperature,
            max_tokens=200,
        )
    except Exception:
        errors_total.labels(type="classifier").inc()
        logger.exception("classifier_failed", model=model, fallback="search_docs")
        return {
            "pending_action": "search_docs",
            "metadata": {
                "classifier_reason": "classifier_failure_fallback",
                "classifier_confidence": "low",
                "classifier_model": model,
            },
        }

    effective_action = parsed.action
    if parsed.confidence == "low":
        effective_action = settings.agent.classifier_low_confidence_action
        logger.info(
            "classifier_low_confidence_downgrade",
            predicted_action=parsed.action,
            effective_action=effective_action,
            reason=parsed.reason,
        )

    logger.info(
        "classifier_decided",
        action=effective_action,
        predicted_action=parsed.action,
        confidence=parsed.confidence,
        model=resp.model,
        tokens_in=resp.tokens_input,
        tokens_out=resp.tokens_output,
        latency_ms=resp.latency_ms,
    )

    return {
        "pending_action": effective_action,
        "metadata": {
            "classifier_reason": parsed.reason,
            "classifier_confidence": parsed.confidence,
            # Its own fields: in `generation_*`, a turn whose answer took no
            # model call, or failed, reported the classifier's usage as the
            # answer's.
            "classifier_model": resp.model,
            "classifier_tokens_input": resp.tokens_input,
            "classifier_tokens_output": resp.tokens_output,
            "classifier_latency_ms": resp.latency_ms,
        },
    }
