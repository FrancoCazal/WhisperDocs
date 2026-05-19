"""Prometheus metrics. Singletons; import wherever needed."""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

agent_turn_duration_seconds = Histogram(
    "whisperdocs_agent_turn_duration_seconds",
    "End-to-end agent turn latency",
    labelnames=("action",),
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0),
)

llm_tokens_total = Counter(
    "whisperdocs_llm_tokens_total",
    "LLM tokens consumed",
    labelnames=("model", "direction"),
)

retrieval_chunks = Histogram(
    "whisperdocs_retrieval_chunks",
    "Chunks returned per retrieval",
    buckets=(0, 1, 3, 5, 8, 13, 20),
)

retrieval_top_relevance = Histogram(
    "whisperdocs_retrieval_top_relevance",
    "Top relevance score per retrieval",
    buckets=(0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

errors_total = Counter(
    "whisperdocs_errors_total",
    "Errors by type",
    labelnames=("type",),
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
