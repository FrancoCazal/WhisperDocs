"""Markdown rendering for EvalReport, and the thresholds a run must meet."""

import json

from app.eval.schemas import AggregateMetrics, EvalReport

# What an old report did not measure: never a 0, which would read as one.
MISSING = "—"


def _pct(value: float | None) -> str:
    return MISSING if value is None else f"{value:.2%}"


def _ms(value: float | None) -> str:
    return MISSING if value is None else f"{value:.0f} ms"


def gate_failures(
    metrics: AggregateMetrics, *, min_accuracy: float, max_hallucination: float
) -> list[str]:
    """The thresholds a run missed, one line each; none when it met them.
    The eval workflow fails a run on any (audit E17). A minimum of 0 and a
    maximum of 1 can never be missed."""
    failures: list[str] = []
    if metrics.accuracy < min_accuracy:
        failures.append(f"accuracy {metrics.accuracy:.1%} is below {min_accuracy:.1%}")
    if metrics.hallucination_rate > max_hallucination:
        failures.append(
            f"hallucination rate {metrics.hallucination_rate:.1%} is above {max_hallucination:.1%}"
        )
    return failures


def diff_failures(
    prev: EvalReport,
    curr: EvalReport,
    *,
    max_accuracy_drop: float,
    max_hallucination_rise: float,
) -> list[str]:
    """How the current run is worse than the previous one past a threshold,
    one line each; none when it is not. The prompt gate fails a pull request
    on any (audit M-4). Two runs over different items are not compared: their
    difference would measure the questions. A threshold of 1 can never be
    passed."""
    if _items(prev) != _items(curr):
        return [
            f"the runs answered different items ({len(prev.results)} and "
            f"{len(curr.results)}), so they are not compared"
        ]
    failures: list[str] = []
    # Rounded: 0.8 - 0.7 is a hair over 0.1 in floating point.
    drop = round(prev.metrics.accuracy - curr.metrics.accuracy, 9)
    if drop > max_accuracy_drop:
        failures.append(
            f"accuracy dropped {drop * 100:.1f} points, from {prev.metrics.accuracy:.1%} to "
            f"{curr.metrics.accuracy:.1%}, more than {max_accuracy_drop * 100:.1f}"
        )
    rise = round(curr.metrics.hallucination_rate - prev.metrics.hallucination_rate, 9)
    if rise > max_hallucination_rise:
        failures.append(
            f"hallucination rate rose {rise * 100:.1f} points, from "
            f"{prev.metrics.hallucination_rate:.1%} to {curr.metrics.hallucination_rate:.1%}, "
            f"more than {max_hallucination_rise * 100:.1f}"
        )
    return failures


def _items(report: EvalReport) -> dict[int, str]:
    return {r.item_id: r.question for r in report.results}


def _version(value: str | None) -> str:
    return MISSING if value is None else f"`{value}`"


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
    lines.append(
        f"- **Prompts**: {_version(report.prompt_version)} "
        f"(judge {_version(report.judge_prompt_version)})"
    )
    lines.append(f"- **Dataset size**: {report.dataset_size}")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Accuracy | {m.accuracy:.2%} |")
    lines.append(f"| Hallucination rate | {m.hallucination_rate:.2%} |")
    lines.append(f"| Retrieval precision@k | {m.avg_retrieval_precision:.2%} |")
    lines.append(f"| Retrieval recall@k | {_pct(m.measured('avg_retrieval_recall'))} |")
    lines.append(f"| Latency avg | {m.avg_latency_ms:.0f} ms |")
    lines.append(f"| Latency p50 | {_ms(m.measured('latency_p50_ms'))} |")
    lines.append(f"| Latency p95 | {_ms(m.measured('latency_p95_ms'))} |")
    lines.append(f"| Avg tokens in / out | {m.avg_tokens_input:.0f} / {m.avg_tokens_output:.0f} |")
    lines.append(f"| Total tokens in / out | {m.total_tokens_input} / {m.total_tokens_output} |")
    lines.append(
        f"| Generation cost (USD, RAG model only; excludes judge and embeddings) "
        f"| ${m.estimated_cost_usd:.4f} |"
    )
    judge_errors = m.measured("judge_errors")
    lines.append(
        f"| Items with a failed judge call | {MISSING if judge_errors is None else judge_errors} |"
    )
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
            reason = (
                r.correctness.reasoning if not r.correctness.passed else r.hallucination.reasoning
            )
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
            f"| {name} | {b.get('total', 0)} | {_pct(b.get('accuracy'))} "
            f"| {_pct(b.get('hallucination_rate'))} "
            f"| {_pct(b.get('avg_retrieval_precision'))} "
            f"| {_pct(b.get('avg_retrieval_recall'))} |"
        )
    return out


def _truncate(text: str, max_chars: int) -> str:
    text = text.replace("|", "\\|").replace("\n", " ").strip()
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def diff_metrics(prev: AggregateMetrics, curr: AggregateMetrics) -> str:
    lines = ["| Metric | Previous | Current | Delta |", "|--------|----------|---------|-------|"]
    rows = [
        ("Accuracy", "accuracy", "pct"),
        ("Hallucination rate", "hallucination_rate", "pct"),
        ("Precision@k", "avg_retrieval_precision", "pct"),
        ("Recall@k", "avg_retrieval_recall", "pct"),
        ("Latency avg (ms)", "avg_latency_ms", "num"),
        ("Latency p50 (ms)", "latency_p50_ms", "num"),
        ("Latency p95 (ms)", "latency_p95_ms", "num"),
        ("Cost (USD)", "estimated_cost_usd", "cost"),
    ]
    # A metric one of the two reports did not measure has no delta.
    value_fmt = {"pct": "{:.2%}", "cost": "${:.4f}", "num": "{:.0f}"}
    delta_fmt = {"pct": "{:+.2%}", "cost": "{:+.4f}", "num": "{:+.0f}"}
    for name, field, kind in rows:
        p, c = prev.measured(field), curr.measured(field)
        shown = [MISSING if v is None else value_fmt[kind].format(v) for v in (p, c)]
        delta = MISSING if p is None or c is None else delta_fmt[kind].format(c - p)
        lines.append(f"| {name} | {shown[0]} | {shown[1]} | {delta} |")
    return "\n".join(lines)


def render_diff_markdown(prev: EvalReport, curr: EvalReport) -> str:
    lines: list[str] = []
    lines.append(f"# Eval Diff — `{prev.run_id[:8]}` → `{curr.run_id[:8]}`")
    lines.append("")
    for label, report in (("Previous", prev), ("Current", curr)):
        lines.append(
            f"- **{label}**: {report.timestamp.isoformat()} (model `{report.model}`, "
            f"prompts {_version(report.prompt_version)}, n={report.dataset_size})"
        )
    lines.append("")
    judges = (prev.judge_prompt_version, curr.judge_prompt_version)
    if None not in judges and judges[0] != judges[1]:
        lines.append(
            f"> The judge's prompts changed ({_version(judges[0])} → {_version(judges[1])}): "
            "part of a metric's change can be the judge's."
        )
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
