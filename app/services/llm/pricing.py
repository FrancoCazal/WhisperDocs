"""Per-model pricing for cost estimation. USD per 1M tokens."""

from dataclasses import dataclass

from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Pricing:
    input_per_1m: float
    output_per_1m: float


# Static table: one row per model the app can be configured with. Update it
# when a default model changes; `tests/test_llm_models.py` fails if a default
# has no row here.
_PRICING: dict[str, Pricing] = {
    # Gemini on Vertex AI. Standard paid tier, <=200K context.
    # gemini-3.8-flash: introductory rate through 2026-12-31, from third-party
    # aggregators — confirm on cloud.google.com/vertex-ai/generative-ai/pricing.
    # Reported to double on 2027-01-01.
    "gemini-3.8-flash": Pricing(input_per_1m=0.75, output_per_1m=3.75),
    # Claude on Bedrock, Converse API (us cross-region inference profiles).
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": Pricing(3.00, 15.00),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": Pricing(1.00, 5.00),
    # Claude Sonnet 5.5, Sonnet 4.5's replacement. From the rate card of its
    # Bedrock Marketplace offer (us-east-1, read on 2026-11-08): global
    # $2/$10, geographic profiles such as `us.` $2.20/$11.
    "global.anthropic.claude-sonnet-5-5": Pricing(2.00, 10.00),
    "us.anthropic.claude-sonnet-5-5": Pricing(2.20, 11.00),
    # OpenAI embeddings: input only. From the model's page on
    # developers.openai.com, read on 2026-11-08.
    "text-embedding-3-small": Pricing(0.02, 0.00),
}

_FALLBACK = Pricing(input_per_1m=1.00, output_per_1m=5.00)

_warned_unknown: set[str] = set()


def has_pricing(model: str) -> bool:
    return model in _PRICING


def estimate_cost_usd(model: str | None, tokens_input: int, tokens_output: int) -> float:
    """Return USD cost for a single LLM call. Falls back to a generic estimate."""
    if not model or not (tokens_input or tokens_output):
        # No call was billed (a turn answered without the model, or a failed
        # call): no lookup, and no unknown-model warning for placeholders.
        return 0.0

    pricing = _PRICING.get(model)
    if pricing is None:
        if model not in _warned_unknown:
            _warned_unknown.add(model)
            logger.warning(
                "pricing_unknown_model",
                model=model,
                fallback_input_per_1m=_FALLBACK.input_per_1m,
                fallback_output_per_1m=_FALLBACK.output_per_1m,
            )
        pricing = _FALLBACK

    cost_in = (tokens_input / 1_000_000) * pricing.input_per_1m
    cost_out = (tokens_output / 1_000_000) * pricing.output_per_1m
    return round(cost_in + cost_out, 6)
