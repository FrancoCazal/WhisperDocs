from app.config import settings
from app.services.llm.base import LLMError, LLMProvider, LLMResponse
from app.services.llm.bedrock_provider import BedrockProvider
from app.services.llm.gemini_provider import GeminiProvider

__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "GeminiProvider",
    "BedrockProvider",
    "llm_provider_for_model",
    "llm_service_for",
]


_gemini_instance: GeminiProvider | None = None
_bedrock_instance: BedrockProvider | None = None


def llm_provider_for_model(model: str) -> LLMProvider:
    """Dispatch provider by model ID convention."""
    global _gemini_instance, _bedrock_instance

    if model.startswith("gemini-") or model.startswith("models/gemini-"):
        if _gemini_instance is None:
            _gemini_instance = GeminiProvider()
        return _gemini_instance

    # Bedrock Claude: on-demand `anthropic.claude-...` or inference-profile `us.`/`eu.`/`apac.`
    if "anthropic" in model or model.startswith(("us.", "eu.", "apac.", "us-gov.")):
        if _bedrock_instance is None:
            _bedrock_instance = BedrockProvider()
        return _bedrock_instance

    raise ValueError(f"Unknown model provider for: {model!r}")


# Node name → agent settings field suffix
_NODE_ALIAS = {
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
    """
    suffix = _NODE_ALIAS.get(node_name)
    if suffix is None:
        raise ValueError(f"Unknown agent node: {node_name!r}")
    model: str = getattr(settings.agent, f"model_{suffix}")
    temperature: float = getattr(settings.agent, f"temp_{suffix}")
    return llm_provider_for_model(model), model, temperature
