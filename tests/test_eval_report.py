"""The eval CLI's markdown: a metric an old report did not save shows as
missing, not as a 0 that reads as a measurement (frontend review BE-5).
"""

from __future__ import annotations

from app.eval.report import diff_metrics, render_diff_markdown, render_markdown
from app.eval.schemas import EvalReport


def _metrics(**overrides: object) -> dict:
    metrics: dict = {
        "total": 2,
        "accuracy": 0.5,
        "hallucination_rate": 0.0,
        "avg_retrieval_precision": 0.75,
        "avg_retrieval_recall": 0.6,
        "avg_latency_ms": 1500.0,
        "latency_p50_ms": 1400.0,
        "latency_p95_ms": 2100.0,
        "avg_tokens_input": 900.0,
        "avg_tokens_output": 80.0,
        "total_tokens_input": 1800,
        "total_tokens_output": 160,
        "estimated_cost_usd": 0.0123,
        "judge_errors": 1,
        "by_difficulty": {},
        "by_category": {
            "in_scope": {
                "total": 2,
                "accuracy": 0.5,
                "hallucination_rate": 0.0,
                "avg_retrieval_precision": 0.75,
                "avg_retrieval_recall": 0.6,
            }
        },
    }
    metrics.update(overrides)
    return metrics


def _report(run_id: str, metrics: dict) -> EvalReport:
    # Read from JSON, as the CLI reads a saved run: what the file lacks stays unset.
    return EvalReport.model_validate(
        {
            "run_id": run_id,
            "timestamp": "2026-11-07T00:00:00Z",
            "dataset_size": 2,
            "model": "test-model",
            "metrics": metrics,
            "results": [],
        }
    )


NEW = _report("11111111-new", _metrics())
# What the April runs saved: no recall, no latency percentiles, no judge
# errors, and buckets without precision or recall.
OLD_METRICS = _metrics(
    by_category={"in_scope": {"total": 2, "accuracy": 0.5, "hallucination_rate": 0.0}}
)
for name in ("avg_retrieval_recall", "latency_p50_ms", "latency_p95_ms", "judge_errors"):
    del OLD_METRICS[name]
OLD = _report("00000000-old", OLD_METRICS)


def _row(markdown: str, label: str) -> str:
    return next(line for line in markdown.splitlines() if line.startswith(f"| {label} |"))


def test_a_metric_one_report_lacks_has_no_delta() -> None:
    diff = diff_metrics(OLD.metrics, NEW.metrics)
    assert _row(diff, "Recall@k") == "| Recall@k | — | 60.00% | — |"
    assert _row(diff, "Latency p50 (ms)") == "| Latency p50 (ms) | — | 1400 | — |"
    assert _row(diff, "Latency p95 (ms)") == "| Latency p95 (ms) | — | 2100 | — |"
    # What both saved still compares.
    assert _row(diff, "Accuracy") == "| Accuracy | 50.00% | 50.00% | +0.00% |"
    assert _row(diff, "Cost (USD)") == "| Cost (USD) | $0.0123 | $0.0123 | +0.0000 |"


def test_two_reports_that_measured_everything_compare_everything() -> None:
    faster = _report("22222222-new", _metrics(latency_p95_ms=1800.0, avg_retrieval_recall=0.7))
    diff = diff_metrics(NEW.metrics, faster.metrics)
    assert _row(diff, "Latency p95 (ms)") == "| Latency p95 (ms) | 2100 | 1800 | -300 |"
    assert _row(diff, "Recall@k") == "| Recall@k | 60.00% | 70.00% | +10.00% |"
    assert "—" not in diff


def test_a_measured_zero_is_still_a_zero() -> None:
    zero = _report("33333333-new", _metrics(avg_retrieval_recall=0.0, judge_errors=0))
    assert _row(diff_metrics(NEW.metrics, zero.metrics), "Recall@k") == (
        "| Recall@k | 60.00% | 0.00% | -60.00% |"
    )
    assert _row(render_markdown(zero), "Items with a failed judge call") == (
        "| Items with a failed judge call | 0 |"
    )


def test_an_old_report_renders_what_it_lacks_as_missing() -> None:
    md = render_markdown(OLD)
    assert _row(md, "Retrieval recall@k") == "| Retrieval recall@k | — |"
    assert _row(md, "Latency p50") == "| Latency p50 | — |"
    assert _row(md, "Latency p95") == "| Latency p95 | — |"
    assert _row(md, "Items with a failed judge call") == "| Items with a failed judge call | — |"
    assert _row(md, "in_scope") == "| in_scope | 2 | 50.00% | 0.00% | — | — |"
    assert "| 0 ms |" not in md


def test_the_diff_report_carries_the_missing_marks() -> None:
    md = render_diff_markdown(OLD, NEW)
    assert _row(md, "Recall@k") == "| Recall@k | — | 60.00% | — |"
    assert "_No regressions detected._" in md
