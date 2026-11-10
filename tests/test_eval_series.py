"""A series of runs of one config (audit E7): `runner run --runs N` saves
each run, then says how far apart they landed and which items flipped, and
gates on the mean."""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime

import pytest

from app.config import settings
from app.eval import runner
from app.eval.metrics import compute_metrics
from app.eval.protocol import mean_interval
from app.eval.report import render_series_markdown
from app.eval.schemas import (
    Category,
    Difficulty,
    EvalReport,
    EvalResult,
    JudgeResult,
    SeriesSummary,
)
from app.eval.series import flaky_items, summarize


def test_the_interval_of_five_runs_uses_t_with_four_degrees():
    low, high = mean_interval([0.88, 0.90, 0.92, 0.94, 0.96])  # type: ignore[misc]
    # sd 0.0316; t(0.975, 4) = 2.776; half = 2.776 * 0.0316 / sqrt(5) = 0.0393.
    assert (round(low, 4), round(high, 4)) == (0.8807, 0.9593)


def test_the_interval_holds_at_its_edges():
    assert mean_interval([0.9]) is None
    assert mean_interval([]) is None
    assert mean_interval([0.9, 0.9, 0.9]) == pytest.approx((0.9, 0.9))
    # Shares: clipped to [0, 1].
    assert mean_interval([1.0, 0.9])[1] == 1.0  # type: ignore[index]
    # Past 30 degrees of freedom, the normal's 1.96.
    values = [0.0, 1.0] * 16
    half = 1.959964 * math.sqrt(sum((v - 0.5) ** 2 for v in values) / 31) / math.sqrt(32)
    assert mean_interval(values) == pytest.approx((0.5 - half, 0.5 + half))


def _result(item_id: int, correct: bool, grounded: bool = True, error: bool = False) -> EvalResult:
    return EvalResult(
        item_id=item_id,
        question=f"question {item_id}",
        expected_answer="a",
        system_answer="a",
        citations=[],
        latency_ms=1000,
        tokens_input=100,
        tokens_output=10,
        correctness=JudgeResult(passed=correct, reasoning="r", error=error),
        hallucination=JudgeResult(passed=grounded, reasoning="r"),
        retrieval_precision=1.0,
        difficulty=Difficulty.EASY,
        category=Category.IN_SCOPE,
    )


def _report(run: int, results: list[EvalResult]) -> EvalReport:
    return EvalReport(
        run_id=f"{run}" * 8 + "-run",
        timestamp=datetime(2026, 11, 9, tzinfo=UTC),
        dataset_size=len(results),
        model="m",
        prompt_version="p" * 12,
        judge_prompt_version="j" * 12,
        metrics=compute_metrics(results, "m"),
        results=results,
    )


def _runs() -> list[EvalReport]:
    # Item 1 always passes. Item 2 passes twice in three. Item 3 is grounded
    # once in three. Item 4's judge failed once, and passed the other two.
    return [
        _report(1, [_result(1, True), _result(2, True), _result(3, True), _result(4, True)]),
        _report(
            2,
            [
                _result(1, True),
                _result(2, False),
                _result(3, True, grounded=False),
                _result(4, False, error=True),
            ],
        ),  # fmt: skip
        _report(
            3,
            [
                _result(1, True),
                _result(2, True),
                _result(3, True, grounded=False),
                _result(4, True),
            ],
        ),  # fmt: skip
    ]


def test_a_series_spreads_the_runs_and_names_the_items_that_flip():
    summary = summarize(_runs())

    assert [r.run_id[:8] for r in summary.runs] == ["11111111", "22222222", "33333333"]
    assert [r.accuracy for r in summary.runs] == [1.0, 0.5, 1.0]
    assert summary.accuracy.mean == pytest.approx(2.5 / 3)
    assert (summary.accuracy.min, summary.accuracy.max) == (0.5, 1.0)
    assert summary.accuracy.low is not None and summary.accuracy.low < summary.accuracy.mean
    assert summary.hallucination_rate.mean == pytest.approx(1 / 6)
    flaky = [(f.item_id, f.check, f.passed, f.judged) for f in summary.flaky]
    # Item 4's failed judge call is no verdict: it passed both it got.
    assert flaky == [(2, "correctness", 2, 3), (3, "hallucination", 1, 3)]
    assert summary.prompt_version == "p" * 12


def test_an_item_that_always_fails_is_not_flaky():
    runs = [_report(i, [_result(1, False), _result(2, i == 1)]) for i in (1, 2)]
    assert [(f.item_id, f.check) for f in flaky_items(runs)] == [(2, "correctness")]


def test_one_run_has_no_interval_and_nothing_flaky():
    (only,) = _runs()[:1]
    summary = summarize([only])
    assert summary.accuracy.low is summary.accuracy.high is None
    assert flaky_items([only]) == []
    md = render_series_markdown(summary)
    assert "| Accuracy | 100.0% | 100.0% | 100.0% | — |" in md
    assert "_Every item got the same verdicts in every run._" in md
    with pytest.raises(ValueError):
        summarize([])


def test_the_series_markdown_shows_the_spread_the_runs_and_the_flaky_items():
    md = render_series_markdown(summarize(_runs()))

    assert md.startswith("# Eval Series — 3 runs")
    assert "| Accuracy | 83.3% | 50.0% | 100.0% |" in md
    assert "| `22222222` | 50.0% | 25.0% | 1 |" in md
    assert "| 2 | correctness | 2 of 3 | question 2 |" in md
    assert "| 3 | hallucination | 1 of 3 | question 3 |" in md


# ------------------------------------------------------------------- CLI


def _cli(monkeypatch, capsys, tmp_path, accuracies: list[float], *flags: str) -> tuple[int, str]:
    calls: list[dict] = []

    async def run_eval(**kwargs):
        calls.append(kwargs)
        i = len(calls)
        correct = round(accuracies[i - 1] * 10)
        return _report(i, [_result(n, n < correct) for n in range(10)])

    monkeypatch.setattr(runner, "run_eval", run_eval)
    argv = ["runner", "run", "--results-dir", str(tmp_path), "--no-judge-cache", *flags]
    monkeypatch.setattr(sys, "argv", argv)
    code = 0
    try:
        runner.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    assert all(c["judge_cache"] is False and c["results_dir"] == tmp_path for c in calls)
    return code, capsys.readouterr().err


def test_the_cli_runs_a_series_and_gates_on_the_mean(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(settings.agent, "eval_min_accuracy", 0.8)
    monkeypatch.setattr(settings.agent, "eval_max_hallucination_rate", 0.1)

    # One run under 80 %, the mean over it: the series passes.
    code, err = _cli(monkeypatch, capsys, tmp_path, [0.7, 0.9, 0.9], "--runs", "3")

    assert (code, err) == (0, "")
    (saved,) = tmp_path.glob("series_*.json")
    summary = SeriesSummary.model_validate_json(saved.read_text(encoding="utf-8"))
    assert [r.accuracy for r in summary.runs] == [0.7, 0.9, 0.9]
    assert saved.with_suffix(".md").read_text(encoding="utf-8").startswith("# Eval Series")


def test_a_series_under_the_threshold_fails_on_its_mean(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(settings.agent, "eval_min_accuracy", 0.8)

    code, err = _cli(monkeypatch, capsys, tmp_path, [0.7, 0.8, 0.9, 0.7], "--runs", "4")

    assert code == runner.EXIT_BELOW_THRESHOLD
    assert "Below the thresholds, over 4 runs:" in err
    assert "- mean accuracy 77.5% is below 80.0%" in err


@pytest.mark.parametrize("runs", ["0", "21", "two"])
def test_a_count_of_runs_out_of_range_is_refused(monkeypatch, capsys, tmp_path, runs):
    code, err = _cli(monkeypatch, capsys, tmp_path, [], "--runs", runs)
    assert code == 2
    assert "--runs" in err
