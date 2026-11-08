from app.config import settings
from app.services.llm.base import LLMError, LLMProvider, LLMResponse
from app.services.llm.bedrock_provider import BedrockProvider
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.pricing import has_pricing
from app.services.llm.tracked import TrackedProvider
from app.services.model_status import Role

__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "GeminiProvider",
    "BedrockProvider",
    "llm_provider_for_model",
    "llm_service_for",
    "provider_name_for_model",
    "validate_agent_models",
]


_gemini_instance: GeminiProvider | None = None
_bedrock_instance: BedrockProvider | None = None


def provider_name_for_model(model: str) -> str:
    """Return "gemini" or "bedrock" for a model ID. Raises ValueError otherwise."""
    if model.startswith("gemini-") or model.startswith("models/gemini-"):
        return "gemini"
    # Bedrock Claude: `anthropic.claude-...`, or an inference profile such as
    # `global.anthropic.claude-...` or `us.anthropic.claude-...`.
    if "anthropic" in model or model.startswith(("us.", "eu.", "apac.", "us-gov.")):
        return "bedrock"
    raise ValueError(
        f"Unknown model provider for: {model!r}. Use a Gemini model name or a Bedrock "
        "Claude ID (`us.anthropic.claude-...`); Anthropic API IDs are not routed."
    )


def llm_provider_for_model(model: str) -> LLMProvider:
    """Dispatch provider by model ID convention."""
    global _gemini_instance, _bedrock_instance

    if provider_name_for_model(model) == "gemini":
        if _gemini_instance is None:
            _gemini_instance = GeminiProvider()
        return _gemini_instance

    if _bedrock_instance is None:
        _bedrock_instance = BedrockProvider()
    return _bedrock_instance


# Node name → agent settings field suffix, which is also its role in
# model_status.
_NODE_ALIAS: dict[str, Role] = {
    "classifier": "classifier",
    "query_rewrite": "query_rewrite",
    "rag": "rag_generation",
    "rag_generation": "rag_generation",
    "clarification": "clarification",
    "ask_clarification": "clarification",
    "answer_direct": "answer_direct",
    "summarize": "summarize",
    "eval_judge": "eval_judge",
}


def llm_service_for(node_name: str) -> tuple[LLMProvider, str, float]:
    """Return (provider, model_id, temperature) for the given node.

    Resolves model + temperature from AgentSettings env vars:
      AGENT_MODEL_<SUFFIX>  / AGENT_TEMP_<SUFFIX>

    The provider records each call's outcome under the node's role
    (app/services/model_status.py).
    """
    suffix = _NODE_ALIAS.get(node_name)
    if suffix is None:
        raise ValueError(f"Unknown agent node: {node_name!r}")
    model: str = getattr(settings.agent, f"model_{suffix}")
    temperature: float = getattr(settings.agent, f"temp_{suffix}")
    return TrackedProvider(llm_provider_for_model(model), suffix), model, temperature


def validate_agent_models() -> tuple[list[str], list[str]]:
    """Check every configured node model. Returns (errors, warnings).

    Errors are model IDs no provider routes, which fail on every call.
    Warnings are models without a row in `pricing.py`, whose cost falls back
    to a generic estimate.
    """
    errors: list[str] = []
    warnings: list[str] = []
    for suffix in sorted(set(_NODE_ALIAS.values())):
        model: str = getattr(settings.agent, f"model_{suffix}")
        field = f"AGENT_MODEL_{suffix.upper()}"
        try:
            provider_name_for_model(model)
        except ValueError as exc:
            errors.append(f"{field}: {exc}")
            continue
        if not has_pricing(model):
            warnings.append(f"{field}: no pricing row for {model!r} in pricing.py")
    return errors, warnings
