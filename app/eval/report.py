"""Markdown rendering for EvalReport, and the thresholds a run must meet."""

import json
from collections.abc import Callable
from dataclasses import dataclass

from app.eval.protocol import mcnemar_exact, wilson_interval
from app.eval.schemas import (
    AggregateMetrics,
    EvalReport,
    EvalResult,
    JudgeResult,
    RunConfig,
    SeriesSummary,
)

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
    return threshold_failures(
        metrics.accuracy,
        metrics.hallucination_rate,
        min_accuracy=min_accuracy,
        max_hallucination=max_hallucination,
    )


def threshold_failures(
    accuracy: float,
    hallucination_rate: float,
    *,
    min_accuracy: float,
    max_hallucination: float,
    of: str = "",
) -> list[str]:
    """`gate_failures` on two shares; `of` names them ("mean ")."""
    failures: list[str] = []
    if accuracy < min_accuracy:
        failures.append(f"{of}accuracy {accuracy:.1%} is below {min_accuracy:.1%}")
    if hallucination_rate > max_hallucination:
        failures.append(
            f"{of}hallucination rate {hallucination_rate:.1%} is above {max_hallucination:.1%}"
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
    if (hashes := _dataset_hashes(prev, curr)) is not None:
        return [
            f"the runs read different datasets ({hashes[0]} and {hashes[1]}): the same "
            "questions with other expected answers or sources, so they are not compared"
        ]
    changes = item_changes(prev, curr)
    failures: list[str] = []
    # Rounded: 0.8 - 0.7 is a hair over 0.1 in floating point.
    drop = round(prev.metrics.accuracy - curr.metrics.accuracy, 9)
    if drop > max_accuracy_drop:
        failures.append(
            f"accuracy dropped {drop * 100:.1f} points, from {prev.metrics.accuracy:.1%} to "
            f"{curr.metrics.accuracy:.1%}, more than {max_accuracy_drop * 100:.1f} "
            f"({_flips_text(changes, 'correctness')})"
        )
    rise = round(curr.metrics.hallucination_rate - prev.metrics.hallucination_rate, 9)
    if rise > max_hallucination_rise:
        failures.append(
            f"hallucination rate rose {rise * 100:.1f} points, from "
            f"{prev.metrics.hallucination_rate:.1%} to {curr.metrics.hallucination_rate:.1%}, "
            f"more than {max_hallucination_rise * 100:.1f} "
            f"({_flips_text(changes, 'hallucination')})"
        )
    return failures


def _dataset_hashes(prev: EvalReport, curr: EvalReport) -> tuple[str, str] | None:
    """The two runs' dataset hashes when both have one and they differ."""
    if prev.config is None or curr.config is None:
        return None
    pair = (prev.config.dataset_hash, curr.config.dataset_hash)
    return pair if pair[0] != pair[1] else None


# The two verdicts of an item; for hallucination, `passed` means grounded.
CHECKS: dict[str, Callable[[EvalResult], JudgeResult]] = {
    "correctness": lambda r: r.correctness,
    "hallucination": lambda r: r.hallucination,
}


@dataclass(frozen=True)
class ItemChange:
    item_id: int
    check: str
    question: str
    # The current run's reasoning: why the item passes or fails now.
    reasoning: str
    # True when it fails now and passed before.
    regression: bool


def item_changes(prev: EvalReport, curr: EvalReport) -> list[ItemChange]:
    """The verdicts that flipped between the runs, over the items both
    answered (same id, same question). A failed judge call is no verdict, so
    it is no flip either."""
    before = {r.item_id: r for r in prev.results}
    changes: list[ItemChange] = []
    for r in curr.results:
        p = before.get(r.item_id)
        if p is None or p.question != r.question:
            continue
        for check, verdict in CHECKS.items():
            was, now = verdict(p), verdict(r)
            if was.error or now.error or was.passed == now.passed:
                continue
            changes.append(ItemChange(r.item_id, check, r.question, now.reasoning, was.passed))
    return changes


def flips(changes: list[ItemChange], check: str) -> tuple[int, int]:
    """Regressions and fixes of one check."""
    worse = sum(1 for c in changes if c.check == check and c.regression)
    better = sum(1 for c in changes if c.check == check and not c.regression)
    return worse, better


def _flips_text(changes: list[ItemChange], check: str) -> str:
    worse, better = flips(changes, check)
    return (
        f"{worse} {'item' if worse == 1 else 'items'} worse, {better} better, "
        f"McNemar p {_p(mcnemar_exact(worse, better))}"
    )


def _p(value: float) -> str:
    return "< 0.001" if value < 0.001 else f"= {value:.3f}"


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
    if report.judge_cache_hits is not None:
        lines.append(f"- **Judge verdicts from the cache**: {report.judge_cache_hits}")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Accuracy | {m.accuracy:.2%} |")
    lines.append(f"| Accuracy, 95% interval | {_interval(m.accuracy, m.total)} |")
    lines.append(f"| Hallucination rate | {m.hallucination_rate:.2%} |")
    lines.append(
        f"| Hallucination rate, 95% interval | {_interval(m.hallucination_rate, m.total)} |"
    )
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
    lines.append("## Run config")
    lines.append("")
    lines.extend(_config_lines(report.config))
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


def _interval(share: float, n: int) -> str:
    """The Wilson interval of a share of n items. The report saves the share
    rounded, so the count is the nearest whole number."""
    low, high = wilson_interval(round(share * n), n)
    return f"{low:.1%} – {high:.1%}"


def _config_lines(config: RunConfig | None) -> list[str]:
    if config is None:
        return ["_This report is older than the run config._"]
    return [
        "| Setting | Value |",
        "|---------|-------|",
        *(f"| {name} | {_setting(value)} |" for name, value in config.model_dump().items()),
    ]


def _setting(value: object) -> str:
    # None: a setting that does not apply, such as a model's default sampling
    # or a step that is off.
    return MISSING if value is None else f"`{value}`"


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
    warnings = comparability_warnings(prev, curr)
    for warning in warnings:
        lines.append(f"> {warning}")
    if warnings:
        lines.append("")
    lines.append("## What changed")
    lines.append("")
    lines.extend(_changed_lines(prev, curr))
    lines.append("")
    lines.append("## Metric deltas")
    lines.append("")
    lines.append(diff_metrics(prev.metrics, curr.metrics))
    lines.append("")
    changes = item_changes(prev, curr)
    lines.append("## Per-item changes")
    lines.append("")
    lines.append(
        "Over the items both runs answered. McNemar's exact test reads only the items that "
        "flipped: p < 0.05 says the flips lean one way more than chance would."
    )
    lines.append("")
    lines.append("| Check | Worse | Better | McNemar p |")
    lines.append("|-------|-------|--------|-----------|")
    for check in CHECKS:
        worse, better = flips(changes, check)
        p_value = _p(mcnemar_exact(worse, better)).removeprefix("= ")
        lines.append(f"| {check} | {worse} | {better} | {p_value} |")
    lines.append("")
    lines.append("### Regressions")
    lines.append("")
    lines.extend(_change_table([c for c in changes if c.regression], "_No regressions detected._"))
    lines.append("")
    lines.append("### Fixes")
    lines.append("")
    lines.extend(_change_table([c for c in changes if not c.regression], "_No fixes._"))
    lines.append("")
    return "\n".join(lines)


def comparability_warnings(prev: EvalReport, curr: EvalReport) -> list[str]:
    """What makes a difference between the runs hard to read, one line each."""
    warnings: list[str] = []
    if (hashes := _dataset_hashes(prev, curr)) is not None:
        warnings.append(
            f"The runs read different datasets ({hashes[0]} → {hashes[1]}): a difference can "
            "be the dataset's."
        )
    judges = (prev.judge_prompt_version, curr.judge_prompt_version)
    if None not in judges and judges[0] != judges[1]:
        warnings.append(
            f"The judge's prompts changed ({_version(judges[0])} → {_version(judges[1])}): "
            "part of a metric's change can be the judge's."
        )
    if prev.config is not None and curr.config is not None:
        judged = [
            name
            for name in ("judge_model", "judge_temperature", "judge_reasoning")
            if getattr(prev.config, name) != getattr(curr.config, name)
        ]
        if judged:
            warnings.append(
                f"The judge changed ({', '.join(judged)}): part of a metric's change can be "
                "the judge's."
            )
    for label, report in (("previous", prev), ("current", curr)):
        if report.config is not None and report.config.git_dirty:
            warnings.append(
                f"The {label} run had uncommitted changes on top of "
                f"{_short_sha(report.config.git_sha)}: no commit holds its code."
            )
    if prev.config is None or curr.config is None:
        warnings.append(
            "A report is older than the run config: past the model and the prompts, "
            "what else changed is unknown."
        )
    return warnings


def _short_sha(sha: str | None) -> str:
    return MISSING if sha is None else f"`{sha[:12]}`"


def _changed_lines(prev: EvalReport, curr: EvalReport) -> list[str]:
    """Each recorded setting that differs between the runs, as a table."""
    rows: list[tuple[str, object, object]] = [
        (name, getattr(prev, name), getattr(curr, name))
        for name in ("model", "prompt_version", "judge_prompt_version")
    ]
    if prev.config is not None and curr.config is not None:
        before, after = prev.config.model_dump(), curr.config.model_dump()
        rows.extend((name, before[name], after[name]) for name in before)
    changed = [(name, a, b) for name, a, b in rows if a != b]
    if not changed:
        return ["_Nothing recorded differs._"]
    return [
        "| Setting | Previous | Current |",
        "|---------|----------|---------|",
        *(f"| {name} | {_setting(a)} | {_setting(b)} |" for name, a, b in changed),
    ]


def _change_table(changes: list[ItemChange], empty: str) -> list[str]:
    if not changes:
        return [empty]
    return [
        "| ID | Check | Question | Reason now |",
        "|----|-------|----------|------------|",
        *(
            f"| {c.item_id} | {c.check} | {_truncate(c.question, 60)} "
            f"| {_truncate(c.reasoning, 80)} |"
            for c in changes
        ),
    ]


def render_series_markdown(summary: SeriesSummary) -> str:
    n = len(summary.runs)
    lines = [f"# Eval Series — {n} {'run' if n == 1 else 'runs'}", ""]
    lines.append(f"- **Timestamp**: {summary.timestamp.isoformat()}")
    lines.append(f"- **Model**: `{summary.model}`")
    lines.append(
        f"- **Prompts**: {_version(summary.prompt_version)} "
        f"(judge {_version(summary.judge_prompt_version)})"
    )
    lines.append(f"- **Dataset size**: {summary.dataset_size}")
    if summary.config is not None:
        lines.append(
            f"- **Commit**: {_setting(summary.config.git_sha)}"
            f"{' (uncommitted changes)' if summary.config.git_dirty else ''}, "
            f"dataset {_setting(summary.config.dataset_hash)}"
        )
    lines += ["", "## Spread", ""]
    lines.append(
        "The 95% interval is the t interval of the mean over the runs: how far the mean of "
        "this many runs of the config can be from the mean of many. It measures the noise "
        "between runs, not the sampling of the items, which each report's Wilson interval "
        "covers."
    )
    lines += ["", "| Metric | Mean | Min | Max | 95% interval of the mean |"]
    lines.append("|--------|------|-----|-----|--------------------------|")
    for label, spread in (
        ("Accuracy", summary.accuracy),
        ("Hallucination rate", summary.hallucination_rate),
    ):
        interval = (
            MISSING
            if spread.low is None or spread.high is None
            else f"{spread.low:.1%} – {spread.high:.1%}"
        )
        lines.append(
            f"| {label} | {spread.mean:.1%} | {spread.min:.1%} | {spread.max:.1%} | {interval} |"
        )
    lines += ["", "## Runs", ""]
    lines.append("| Run | Accuracy | Hallucination rate | Items with a failed judge call |")
    lines.append("|-----|----------|--------------------|--------------------------------|")
    for run in summary.runs:
        lines.append(
            f"| `{run.run_id[:8]}` | {run.accuracy:.1%} | {run.hallucination_rate:.1%} "
            f"| {run.judge_errors} |"
        )
    lines += ["", "## Flaky items", ""]
    if not summary.flaky:
        lines.append("_Every item got the same verdicts in every run._")
    else:
        lines.append("Items whose verdict on a check differed between runs.")
        lines += ["", "| ID | Check | Passed | Question |", "|----|-------|--------|----------|"]
        for item in summary.flaky:
            lines.append(
                f"| {item.item_id} | {item.check} | {item.passed} of {item.judged} "
                f"| {_truncate(item.question, 60)} |"
            )
    lines.append("")
    return "\n".join(lines)
