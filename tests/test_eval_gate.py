"""The eval CLI fails a run that misses a threshold (audit E17).

The eval workflow runs the CLI, so a regression fails the job instead of
leaving a report nobody reads. The thresholds come from settings; the flags
override them.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import pytest

from app.config import settings
from app.eval import runner
from app.eval.report import gate_failures
from app.eval.schemas import AggregateMetrics, EvalReport


def _metrics(accuracy: float, hallucination_rate: float) -> AggregateMetrics:
    return AggregateMetrics(
        total=50,
        accuracy=accuracy,
        hallucination_rate=hallucination_rate,
        avg_retrieval_precision=0.8,
        avg_latency_ms=1500.0,
        avg_tokens_input=900.0,
        avg_tokens_output=80.0,
        total_tokens_input=45_000,
        total_tokens_output=4_000,
        estimated_cost_usd=0.5,
        by_difficulty={},
        by_category={},
    )


def test_a_run_at_its_thresholds_passes():
    assert gate_failures(_metrics(0.8, 0.1), min_accuracy=0.8, max_hallucination=0.1) == []


def test_a_run_past_either_threshold_fails_with_both_named():
    failures = gate_failures(_metrics(0.78, 0.12), min_accuracy=0.8, max_hallucination=0.1)

    assert failures == [
        "accuracy 78.0% is below 80.0%",
        "hallucination rate 12.0% is above 10.0%",
    ]


def test_zero_and_one_turn_the_thresholds_off():
    assert gate_failures(_metrics(0.0, 1.0), min_accuracy=0.0, max_hallucination=1.0) == []


def _run(monkeypatch, capsys, metrics: AggregateMetrics, *flags: str) -> tuple[int, str]:
    async def run_eval(**_kwargs):
        return EvalReport(
            run_id="r1",
            timestamp=datetime.now(UTC),
            dataset_size=metrics.total,
            model="m",
            metrics=metrics,
            results=[],
        )

    monkeypatch.setattr(runner, "run_eval", run_eval)
    monkeypatch.setattr(sys, "argv", ["runner", "run", *flags])
    code = 0
    try:
        runner.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    return code, capsys.readouterr().err


def test_the_cli_exits_3_below_the_configured_accuracy(monkeypatch, capsys):
    monkeypatch.setattr(settings.agent, "eval_min_accuracy", 0.9)
    monkeypatch.setattr(settings.agent, "eval_max_hallucination_rate", 0.1)

    code, err = _run(monkeypatch, capsys, _metrics(0.88, 0.02))

    assert code == runner.EXIT_BELOW_THRESHOLD
    assert "accuracy 88.0% is below 90.0%" in err


def test_a_flag_overrides_the_configured_threshold(monkeypatch, capsys):
    monkeypatch.setattr(settings.agent, "eval_min_accuracy", 0.9)
    monkeypatch.setattr(settings.agent, "eval_max_hallucination_rate", 0.1)

    code, err = _run(monkeypatch, capsys, _metrics(0.88, 0.02), "--min-accuracy", "0.85")

    assert code == 0
    assert err == ""


@pytest.mark.parametrize("value", ["1.5", "-0.1", "abc"])
def test_a_threshold_outside_0_to_1_is_refused(monkeypatch, capsys, value):
    code, err = _run(monkeypatch, capsys, _metrics(0.9, 0.0), "--max-hallucination", value)

    assert code == 2
    assert "--max-hallucination" in err
