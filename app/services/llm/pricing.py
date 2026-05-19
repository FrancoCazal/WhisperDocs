"""Per-model pricing for cost estimation. USD per 1M tokens."""

from dataclasses import dataclass

from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Pricing:
    input_per_1m: float
    output_per_1m: float


# Static table. Update when providers change pricing or new models added.
_PRICING: dict[str, Pricing] = {
    # Gemini (Vertex AI)
    "gemini-2.5-flash": Pricing(input_per_1m=0.30, output_per_1m=2.50),
    "gemini-2.5-pro": Pricing(input_per_1m=1.25, output_per_1m=10.00),
    "gemini-1.5-flash": Pricing(input_per_1m=0.075, output_per_1m=0.30),
    "gemini-1.5-pro": Pricing(input_per_1m=1.25, output_per_1m=5.00),
    # Claude on Bedrock (us inference profiles)
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": Pricing(3.00, 15.00),
    "us.anthropic.claude-sonnet-4-20250514-v1:0": Pricing(3.00, 15.00),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": Pricing(1.00, 5.00),
    "us.anthropic.claude-opus-4-7-20251201-v1:0": Pricing(15.00, 75.00),
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": Pricing(3.00, 15.00),
}

_FALLBACK = Pricing(input_per_1m=1.00, output_per_1m=5.00)


def estimate_cost_usd(model: str | None, tokens_input: int, tokens_output: int) -> float:
    """Return USD cost for a single LLM call. Falls back to a generic estimate."""
    if not model:
        return 0.0

    pricing = _PRICING.get(model)
    if pricing is None:
        logger.debug("pricing_unknown_model", model=model)
        pricing = _FALLBACK

    cost_in = (tokens_input / 1_000_000) * pricing.input_per_1m
    cost_out = (tokens_output / 1_000_000) * pricing.output_per_1m
    return round(cost_in + cost_out, 6)
