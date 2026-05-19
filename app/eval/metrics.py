from collections import defaultdict

from app.eval.schemas import AggregateMetrics, EvalResult
from app.services.llm.pricing import estimate_cost_usd


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, int(pct * len(sorted_vals)))
    return float(sorted_vals[idx])


def _bucket_metrics(results: list[EvalResult]) -> dict:
    if not results:
        return {
            "total": 0,
            "accuracy": 0.0,
            "hallucination_rate": 0.0,
            "avg_retrieval_precision": 0.0,
            "avg_retrieval_recall": 0.0,
        }
    correct = sum(1 for r in results if r.correctness.passed)
    hallucinated = sum(1 for r in results if not r.hallucination.passed)
    scoped = [r for r in results if r.retrieval_precision or r.retrieval_recall]
    if scoped:
        avg_p = sum(r.retrieval_precision for r in scoped) / len(scoped)
        avg_r = sum(r.retrieval_recall for r in scoped) / len(scoped)
    else:
        avg_p = 0.0
        avg_r = 0.0
    return {
        "total": len(results),
        "accuracy": round(correct / len(results), 4),
        "hallucination_rate": round(hallucinated / len(results), 4),
        "avg_retrieval_precision": round(avg_p, 4),
        "avg_retrieval_recall": round(avg_r, 4),
    }


def compute_metrics(results: list[EvalResult], model: str) -> AggregateMetrics:
    total = len(results)
    if total == 0:
        raise ValueError("Cannot compute metrics on empty results")

    correct = sum(1 for r in results if r.correctness.passed)
    hallucinated = sum(1 for r in results if not r.hallucination.passed)

    total_tokens_in = sum(r.tokens_input for r in results)
    total_tokens_out = sum(r.tokens_output for r in results)

    cost = estimate_cost_usd(model, total_tokens_in, total_tokens_out)

    scoped = [r for r in results if r.retrieval_precision or r.retrieval_recall]
    if scoped:
        avg_precision = sum(r.retrieval_precision for r in scoped) / len(scoped)
        avg_recall = sum(r.retrieval_recall for r in scoped) / len(scoped)
    else:
        avg_precision = 0.0
        avg_recall = 0.0

    latencies = [float(r.latency_ms) for r in results]

    by_difficulty: dict[str, list[EvalResult]] = defaultdict(list)
    by_category: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_difficulty[r.difficulty.value].append(r)
        by_category[r.category.value].append(r)

    return AggregateMetrics(
        total=total,
        accuracy=round(correct / total, 4),
        hallucination_rate=round(hallucinated / total, 4),
        avg_retrieval_precision=round(avg_precision, 4),
        avg_retrieval_recall=round(avg_recall, 4),
        avg_latency_ms=round(sum(latencies) / total, 1),
        latency_p50_ms=round(_percentile(latencies, 0.50), 1),
        latency_p95_ms=round(_percentile(latencies, 0.95), 1),
        avg_tokens_input=round(total_tokens_in / total, 1),
        avg_tokens_output=round(total_tokens_out / total, 1),
        total_tokens_input=total_tokens_in,
        total_tokens_output=total_tokens_out,
        estimated_cost_usd=round(cost, 6),
        by_difficulty={k: _bucket_metrics(v) for k, v in by_difficulty.items()},
        by_category={k: _bucket_metrics(v) for k, v in by_category.items()},
    )
