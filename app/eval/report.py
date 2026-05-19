"""Markdown rendering for EvalReport."""

import json

from app.eval.schemas import AggregateMetrics, EvalReport


def render_badge_json(report: EvalReport) -> str:
    """Shields.io endpoint format. https://shields.io/endpoint"""
    accuracy = report.metrics.accuracy
    if accuracy >= 0.85:
        color = "brightgreen"
    elif accuracy >= 0.70:
        color = "yellow"
    else:
        color = "red"
    return json.dumps(
        {
            "schemaVersion": 1,
            "label": "eval accuracy",
            "message": f"{accuracy:.0%}",
            "color": color,
        }
    )


def render_markdown(report: EvalReport) -> str:
    m = report.metrics
    lines: list[str] = []
    lines.append(f"# Eval Report — `{report.run_id[:8]}`")
    lines.append("")
    lines.append(f"- **Timestamp**: {report.timestamp.isoformat()}")
    lines.append(f"- **Model**: `{report.model}`")
    lines.append(f"- **Dataset size**: {report.dataset_size}")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Accuracy | {m.accuracy:.2%} |")
    lines.append(f"| Hallucination rate | {m.hallucination_rate:.2%} |")
    lines.append(f"| Retrieval precision@k | {m.avg_retrieval_precision:.2%} |")
    lines.append(f"| Retrieval recall@k | {m.avg_retrieval_recall:.2%} |")
    lines.append(f"| Latency avg | {m.avg_latency_ms:.0f} ms |")
    lines.append(f"| Latency p50 | {m.latency_p50_ms:.0f} ms |")
    lines.append(f"| Latency p95 | {m.latency_p95_ms:.0f} ms |")
    lines.append(f"| Avg tokens in / out | {m.avg_tokens_input:.0f} / {m.avg_tokens_output:.0f} |")
    lines.append(f"| Total tokens in / out | {m.total_tokens_input} / {m.total_tokens_output} |")
    lines.append(f"| Estimated cost (USD) | ${m.estimated_cost_usd:.4f} |")
    lines.append("")
    lines.append("## By difficulty")
    lines.append("")
    lines.extend(_bucket_table(m.by_difficulty))
    lines.append("")
    lines.append("## By category")
    lines.append("")
    lines.extend(_bucket_table(m.by_category))
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    failures = [r for r in report.results if not r.correctness.passed or not r.hallucination.passed]
    if not failures:
        lines.append("_No failures._")
    else:
        lines.append("| ID | Question | Correctness | Hallucination | Reasoning |")
        lines.append("|----|----------|-------------|---------------|-----------|")
        for r in failures:
            corr = "OK" if r.correctness.passed else "FAIL"
            hall = "OK" if r.hallucination.passed else "FAIL"
            reason = (r.correctness.reasoning if not r.correctness.passed else r.hallucination.reasoning)
            lines.append(
                f"| {r.item_id} | {_truncate(r.question, 60)} | {corr} | {hall} | {_truncate(reason, 80)} |"
            )
    lines.append("")
    return "\n".join(lines)


def _bucket_table(buckets: dict[str, dict]) -> list[str]:
    if not buckets:
        return ["_(no data)_"]
    out = [
        "| Bucket | n | Accuracy | Halluc. rate | Precision@k | Recall@k |",
        "|--------|---|----------|--------------|-------------|----------|",
    ]
    for name, b in sorted(buckets.items()):
        out.append(
            f"| {name} | {b.get('total', 0)} | {b.get('accuracy', 0):.2%} "
            f"| {b.get('hallucination_rate', 0):.2%} "
            f"| {b.get('avg_retrieval_precision', 0):.2%} "
            f"| {b.get('avg_retrieval_recall', 0):.2%} |"
        )
    return out


def _truncate(text: str, max_chars: int) -> str:
    text = text.replace("|", "\\|").replace("\n", " ").strip()
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def diff_metrics(prev: AggregateMetrics, curr: AggregateMetrics) -> str:
    lines = ["| Metric | Previous | Current | Delta |", "|--------|----------|---------|-------|"]
    rows = [
        ("Accuracy", prev.accuracy, curr.accuracy, "pct"),
        ("Hallucination rate", prev.hallucination_rate, curr.hallucination_rate, "pct"),
        ("Precision@k", prev.avg_retrieval_precision, curr.avg_retrieval_precision, "pct"),
        ("Recall@k", prev.avg_retrieval_recall, curr.avg_retrieval_recall, "pct"),
        ("Latency avg (ms)", prev.avg_latency_ms, curr.avg_latency_ms, "num"),
        ("Latency p50 (ms)", prev.latency_p50_ms, curr.latency_p50_ms, "num"),
        ("Latency p95 (ms)", prev.latency_p95_ms, curr.latency_p95_ms, "num"),
        ("Cost (USD)", prev.estimated_cost_usd, curr.estimated_cost_usd, "cost"),
    ]
    for name, p, c, kind in rows:
        delta = c - p
        if kind == "pct":
            cell = f"| {name} | {p:.2%} | {c:.2%} | {delta:+.2%} |"
        elif kind == "cost":
            cell = f"| {name} | ${p:.4f} | ${c:.4f} | {delta:+.4f} |"
        else:
            cell = f"| {name} | {p:.0f} | {c:.0f} | {delta:+.0f} |"
        lines.append(cell)
    return "\n".join(lines)


def render_diff_markdown(prev: EvalReport, curr: EvalReport) -> str:
    lines: list[str] = []
    lines.append(f"# Eval Diff — `{prev.run_id[:8]}` → `{curr.run_id[:8]}`")
    lines.append("")
    lines.append(f"- **Previous**: {prev.timestamp.isoformat()} (model `{prev.model}`, n={prev.dataset_size})")
    lines.append(f"- **Current**:  {curr.timestamp.isoformat()} (model `{curr.model}`, n={curr.dataset_size})")
    lines.append("")
    lines.append("## Metric deltas")
    lines.append("")
    lines.append(diff_metrics(prev.metrics, curr.metrics))
    lines.append("")
    lines.append("## Per-item regressions")
    lines.append("")
    prev_by_id = {r.item_id: r for r in prev.results}
    regressions = []
    for r in curr.results:
        p = prev_by_id.get(r.item_id)
        if p is None:
            continue
        if p.correctness.passed and not r.correctness.passed:
            regressions.append((r.item_id, "correctness", r.question, r.correctness.reasoning))
        if p.hallucination.passed and not r.hallucination.passed:
            regressions.append((r.item_id, "hallucination", r.question, r.hallucination.reasoning))
    if not regressions:
        lines.append("_No regressions detected._")
    else:
        lines.append("| ID | Kind | Question | Reason |")
        lines.append("|----|------|----------|--------|")
        for item_id, kind, q, reason in regressions:
            lines.append(f"| {item_id} | {kind} | {_truncate(q, 60)} | {_truncate(reason, 80)} |")
    lines.append("")
    return "\n".join(lines)
