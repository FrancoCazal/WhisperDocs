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
    "Highest cosine similarity among the chunks of a retrieval",
    buckets=(0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Every model call, by its role (classifier, rag_generation, ..., embedding):
# outcome is "ok" or the failure's kind (billing, auth, rate_limit, ...).
model_calls_total = Counter(
    "whisperdocs_model_calls_total",
    "Model calls by role, model and outcome",
    labelnames=("role", "model", "outcome"),
)

errors_total = Counter(
    "whisperdocs_errors_total",
    "Errors by type",
    labelnames=("type",),
)

# HTTP rate, errors and duration (audit E13), from `HTTPMetricsMiddleware`.
# `route` is the matched route's template, never the raw path, so its values
# stay few; a path no route matches is "unmatched".
http_requests_total = Counter(
    "whisperdocs_http_requests_total",
    "HTTP requests by method, route template and status code",
    labelnames=("method", "route", "status"),
)

http_request_duration_seconds = Histogram(
    "whisperdocs_http_request_duration_seconds",
    "HTTP request duration until the response's last byte; a stream's covers the whole stream",
    labelnames=("method", "route"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0),
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
