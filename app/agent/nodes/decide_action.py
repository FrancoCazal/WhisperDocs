from app.agent.prompts import CLASSIFIER_SYSTEM_PROMPT
from app.agent.schemas import AgentAction
from app.agent.state import AgentState
from app.agent.utils.messages import build_prompt_messages
from app.config import settings
from app.logging_config import get_logger
from app.services.llm import llm_service_for

logger = get_logger(__name__)


async def decide_action_node(state: AgentState) -> dict:
    """LLM classifier with structured outputs → routes to one terminal node.

    Low-confidence predictions are downgraded to `classifier_low_confidence_action`
    (default `ask_clarification`) to avoid silently routing uncertain calls to
    search_docs or answer_direct. Exception fallback also goes to clarification:
    safer than the previous answer_direct fallback, which could hallucinate.
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
        logger.exception("classifier_failed", model=model, fallback="ask_clarification")
        return {
            "pending_action": "ask_clarification",
            "metadata": {
                "classifier_reason": "classifier_failure_fallback",
                "classifier_confidence": "low",
                "generation_model": model,
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
            "generation_model": resp.model,
            "generation_tokens_input": resp.tokens_input,
            "generation_tokens_output": resp.tokens_output,
            "generation_latency_ms": resp.latency_ms,
        },
    }
