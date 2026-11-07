"""Eval aggregation (audit E2) and judge-failure handling (review N2)."""

from __future__ import annotations

import pytest

from app.eval.evaluator import EvalAbortedError, _raise_if_judge_unreliable
from app.eval.metrics import compute_metrics
from app.eval.schemas import Category, Difficulty, EvalResult, JudgeResult

OK = JudgeResult(passed=True, reasoning="ok")
WRONG = JudgeResult(passed=False, reasoning="wrong")
JUDGE_DOWN = JudgeResult(passed=False, reasoning="Judge call failed: NotFound", error=True)


def _result(
    item_id: int,
    *,
    category: Category = Category.IN_SCOPE,
    precision: float = 1.0,
    recall: float = 1.0,
    correctness: JudgeResult = OK,
    hallucination: JudgeResult = OK,
) -> EvalResult:
    return EvalResult(
        item_id=item_id,
        question=f"q{item_id}",
        expected_answer="a",
        system_answer="a",
        citations=[],
        latency_ms=100,
        tokens_input=1000,
        tokens_output=100,
        correctness=correctness,
        hallucination=hallucination,
        retrieval_precision=precision,
        retrieval_recall=recall,
        difficulty=Difficulty.EASY,
        category=category,
    )


def test_retrieval_average_counts_in_scope_items_that_retrieved_nothing():
    """The old filter (`precision or recall`) dropped item 2 and reported 1.0."""
    results = [
        _result(1, precision=1.0, recall=1.0),
        _result(2, precision=0.0, recall=0.0),
        _result(3, category=Category.OUT_OF_SCOPE, precision=0.0, recall=0.0),
    ]

    m = compute_metrics(results, model="us.anthropic.claude-sonnet-4-5-20250929-v1:0")

    assert m.avg_retrieval_precision == 0.5
    assert m.avg_retrieval_recall == 0.5
    assert m.by_category["in_scope"]["avg_retrieval_precision"] == 0.5
    assert m.by_category["out_of_scope"]["avg_retrieval_precision"] == 0.0


def test_accuracy_hallucination_and_cost():
    results = [
        _result(1),
        _result(2, correctness=WRONG),
        _result(3, hallucination=WRONG),
        _result(4, category=Category.OUT_OF_SCOPE),
    ]

    m = compute_metrics(results, model="us.anthropic.claude-sonnet-4-5-20250929-v1:0")

    assert m.accuracy == 0.75
    assert m.hallucination_rate == 0.25
    # 4000 in × $3/1M + 400 out × $15/1M
    assert m.estimated_cost_usd == pytest.approx(0.018)
    assert m.judge_errors == 0


def test_judge_errors_are_counted_per_item():
    results = [
        _result(1, correctness=JUDGE_DOWN, hallucination=JUDGE_DOWN),
        _result(2, hallucination=JUDGE_DOWN),
        _result(3),
    ]

    assert compute_metrics(results, model="m").judge_errors == 2


def test_eval_aborts_when_the_judge_is_down():
    results = [_result(i, correctness=JUDGE_DOWN, hallucination=JUDGE_DOWN) for i in range(10)]

    with pytest.raises(EvalAbortedError) as excinfo:
        _raise_if_judge_unreliable(results, judge_errors=10)

    assert excinfo.value.judge_errors == 10
    assert "NotFound" in excinfo.value.first_error
    assert "NotFound" not in str(excinfo.value)  # API detail carries counts only


def test_eval_tolerates_an_isolated_judge_error():
    results = [_result(i) for i in range(19)] + [_result(19, correctness=JUDGE_DOWN)]

    _raise_if_judge_unreliable(results, judge_errors=1)  # 5% <= 10% default
