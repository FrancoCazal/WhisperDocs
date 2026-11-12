"""Does the judge agree with a person? (audit E4): `runner label` draws a blind,
balanced sample of a run's verdicts, and `runner agreement` sets a person's
rulings against the judge's. No model is called."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.eval import runner
from app.eval import schemas as eval_schemas
from app.eval.agreement import (
    _allocate,
    build_labels,
    cohen_kappa,
    measure_agreement,
    read_labels,
    write_labels,
)
from app.eval.metrics import compute_metrics
from app.eval.protocol import context_hash
from app.eval.report import CHECKS, render_agreement_markdown
from app.eval.schemas import (
    Category,
    Difficulty,
    EvalReport,
    EvalResult,
    JudgeResult,
    LabelRow,
)

RUN_ID = "abcdef12-0000-run"


def _result(
    item_id: int,
    *,
    correct: bool = True,
    grounded: bool = True,
    answer: str = "an answer",
    context: bool = True,
    correctness_error: bool = False,
    facts_missing: list[str] | None = None,
) -> EvalResult:
    texts = [f"text {item_id}"]
    return EvalResult(
        item_id=item_id,
        question=f"question {item_id}",
        expected_answer="a",
        system_answer=answer,
        citations=[],
        latency_ms=1,
        tokens_input=1,
        tokens_output=1,
        correctness=JudgeResult(
            reasoning=f"correctness {item_id}",
            passed=correct,
            error=correctness_error,
            facts_missing=facts_missing or [],
        ),
        hallucination=JudgeResult(reasoning=f"grounding {item_id}", passed=grounded),
        retrieval_precision=1.0,
        context_chunk_ids=[item_id] if context else [],
        context_hash=context_hash(texts) if context else context_hash([]),
        difficulty=Difficulty.EASY,
        category=Category.IN_SCOPE,
    )


def _report(results: list[EvalResult], run_id: str = RUN_ID) -> EvalReport:
    return EvalReport(
        run_id=run_id,
        timestamp=datetime(2026, 11, 10, tzinfo=UTC),
        dataset_size=len(results),
        model="m",
        judge_prompt_version="j" * 12,
        metrics=compute_metrics(results, "m"),
        results=results,
    )


def _item(item_id: int, **overrides: object) -> eval_schemas.TestItem:
    fields: dict = {
        "id": item_id,
        "question": f"question {item_id}",
        "expected_answer": "a",
        "difficulty": Difficulty.EASY,
        "category": Category.IN_SCOPE,
        "required_facts": [f"fact of {item_id}"],
    }
    return eval_schemas.TestItem(**{**fields, **overrides})


def _run(n: int = 30):
    """30 items: the judge fails the correctness of every fifth and the
    grounding of items 3 and 6."""
    results = [_result(i, correct=i % 5 != 0, grounded=i not in (3, 6)) for i in range(1, n + 1)]
    items = [_item(r.item_id) for r in results]
    contexts = {i: f"text {i}" for i in range(1, n + 1)}
    return _report(results), items, contexts


# ------------------------------------------------------------------- sampling


def test_the_sample_is_half_of_each_check_and_balanced_by_the_judges_verdict():
    report, items, contexts = _run()

    rows = build_labels(report, items, contexts, n=10, seed=0)

    correctness = [r for r in rows if r.check == "correctness"]
    grounding = [r for r in rows if r.check == "hallucination"]
    assert (len(correctness), len(grounding)) == (5, 5)
    # Of five, three the judge passed and two it failed, though only a sixth
    # of the run's correctness verdicts are fails.
    assert sorted(r.item_id % 5 == 0 for r in correctness) == [False, False, False, True, True]
    assert sorted(r.item_id in (3, 6) for r in grounding) == [False, False, False, True, True]


def test_the_rows_show_what_the_judge_read_and_not_what_it_ruled():
    report, items, contexts = _run()

    rows = build_labels(report, items, contexts, n=500, seed=0)

    assert all(r.human is None and r.run_id == RUN_ID for r in rows)
    assert not {"passed", "reasoning", "error", "facts_missing"} & set(rows[0].model_dump())
    for row in rows:
        blob = row.model_dump_json()
        assert f"correctness {row.item_id}" not in blob
        assert f"grounding {row.item_id}" not in blob
    correctness = next(r for r in rows if r.check == "correctness")
    grounding = next(r for r in rows if r.check == "hallucination")
    assert correctness.required_facts == [f"fact of {correctness.item_id}"]
    assert correctness.expected_answer == "a" and correctness.context is None
    assert grounding.context == [f"text {grounding.item_id}"]
    assert grounding.expected_answer is None and grounding.required_facts == []
    assert "correct" in correctness.ask and "supported" in grounding.ask


def test_the_same_seed_draws_the_same_sample_in_an_order_that_hides_the_strata():
    report, items, contexts = _run()

    first = build_labels(report, items, contexts, n=10, seed=3)

    assert first == build_labels(report, items, contexts, n=10, seed=3)
    assert first != build_labels(report, items, contexts, n=10, seed=4)
    checks = [r.check for r in first]
    assert checks != sorted(checks) and checks != sorted(checks, reverse=True)


def test_a_verdict_the_judge_did_not_give_is_not_sampled():
    results = [
        _result(1, answer="PWNED"),  # fails by rule: no judge call
        _result(2, correctness_error=True),  # the call failed
        _result(3, context=False),  # grounded by rule: no context
        _result(4),
    ]
    items = [_item(1, forbidden=["pwned"]), _item(2), _item(3), _item(4)]
    contexts = {1: "text 1", 2: "text 2", 4: "text 4"}

    rows = build_labels(_report(results), items, contexts, n=500, seed=0)

    seen = sorted((r.check, r.item_id) for r in rows)
    assert seen == [
        ("correctness", 3),
        ("correctness", 4),
        ("hallucination", 1),
        ("hallucination", 2),
        ("hallucination", 4),
    ]


def test_a_chunk_that_changed_since_the_run_leaves_its_grounding_out():
    results = [_result(1), _result(2)]
    contexts = {1: "text 1", 2: "rewritten since the run"}

    rows = build_labels(_report(results), [_item(1), _item(2)], contexts, n=500, seed=0)

    assert [(r.check, r.item_id) for r in rows if r.check == "hallucination"] == [
        ("hallucination", 1)
    ]


def test_asking_for_more_than_there_is_returns_everything():
    report, items, contexts = _run(6)
    assert len(build_labels(report, items, contexts, n=500, seed=0)) == 12


@pytest.mark.parametrize(
    ("total", "sizes", "expected"),
    [
        (10, [3, 20], [3, 7]),
        (4, [10, 10], [2, 2]),
        (5, [10, 10], [3, 2]),
        (5, [0, 0], [0, 0]),
        (5, [0, 9], [0, 5]),
        (50, [10, 10], [10, 10]),
        (0, [4, 4], [0, 0]),
    ],
)
def test_a_share_a_small_group_cannot_take_goes_to_the_others(total, sizes, expected):
    assert _allocate(total, sizes) == expected


# ---------------------------------------------------------------------- kappa


def test_kappa_is_agreement_beyond_chance():
    assert cohen_kappa(10, 0, 0, 10) == 1.0
    # 35 of 50 agree; each side passes 50% and 60%, so chance gives 50%.
    assert cohen_kappa(20, 5, 10, 15) == pytest.approx(0.4)
    # Agreeing as often as two coins would.
    assert cohen_kappa(5, 5, 5, 5) == 0.0
    # Systematically opposed.
    assert cohen_kappa(0, 10, 10, 0) == -1.0


def test_kappa_is_undefined_when_there_is_nothing_to_measure_or_a_side_never_varies():
    assert cohen_kappa(0, 0, 0, 0) is None
    assert cohen_kappa(10, 0, 0, 0) is None


# ------------------------------------------------------------------ agreement


def _label(item_id: int, check: str, human: bool | None, **overrides: object) -> LabelRow:
    fields: dict = {
        "run_id": RUN_ID,
        "item_id": item_id,
        "check": check,
        "ask": "?",
        "question": f"question {item_id}",
        "system_answer": "an answer",
        "human": human,
    }
    return LabelRow(**{**fields, **overrides})


def _judged_run() -> EvalReport:
    return _report(
        [
            _result(1),
            _result(2),
            _result(3, correct=False, grounded=False),
            _result(4, correct=False, facts_missing=["fact x"]),
            _result(5, correctness_error=True),
        ]
    )


def test_agreement_counts_each_kind_of_split_and_keeps_the_disagreements():
    rows = [
        _label(1, "correctness", True),  # the judge passed it, the person too
        _label(2, "correctness", False),  # the judge passed it, the person would not
        _label(3, "correctness", False),  # both fail it
        _label(4, "correctness", True),  # the judge failed it, the person would not
        _label(1, "hallucination", True),
        _label(3, "hallucination", False),
        _label(2, "hallucination", None),  # not ruled on yet
        _label(1, "correctness", True, run_id="another-run"),
        _label(99, "correctness", True),
        _label(2, "correctness", True, question="not the run's question"),
        _label(5, "correctness", True),  # the judge's call failed: nothing to set against
    ]

    summary = measure_agreement(_judged_run(), rows)

    correctness, grounding = summary.checks
    assert correctness.model_dump() == {
        "check": "correctness",
        "labeled": 4,
        "agree": 2,
        "both_pass": 1,
        "judge_pass_human_fail": 1,
        "judge_fail_human_pass": 1,
        "both_fail": 1,
        "kappa": 0.0,
    }
    assert (grounding.labeled, grounding.agree, grounding.kappa) == (2, 2, 1.0)
    assert (summary.unlabeled, summary.unmatched) == (1, 4)
    assert summary.judge_prompt_version == "j" * 12
    assert [
        (d.check, d.item_id, d.judge_passed, d.human_passed) for d in summary.disagreements
    ] == [
        ("correctness", 2, True, False),
        ("correctness", 4, False, True),
    ]
    assert summary.disagreements[1].facts_missing == ["fact x"]
    assert summary.disagreements[1].judge_reasoning == "correctness 4"


def test_a_label_note_travels_with_its_disagreement():
    summary = measure_agreement(
        _judged_run(), [_label(2, "correctness", False, note="it never says L2")]
    )
    assert summary.disagreements[0].note == "it never says L2"


def test_nothing_labeled_measures_nothing():
    summary = measure_agreement(_judged_run(), [_label(1, "correctness", None)])

    assert [c.labeled for c in summary.checks] == [0, 0]
    assert [c.kappa for c in summary.checks] == [None, None]
    assert summary.unlabeled == 1 and summary.disagreements == []


def test_the_markdown_reports_agreement_kappa_and_the_splits():
    rows = [
        _label(1, "correctness", True),
        _label(2, "correctness", False, note="misses a fact"),
        _label(3, "correctness", False),
        _label(4, "correctness", True),
        _label(1, "hallucination", True),
        _label(3, "hallucination", False),
    ]

    markdown = render_agreement_markdown(measure_agreement(_judged_run(), rows))

    assert "# Judge vs Human — `abcdef12`" in markdown
    assert "| correctness | 4 | 2 | 50.0% |" in markdown
    assert "| hallucination | 2 | 2 | 100.0% |" in markdown
    assert "| 0.00 |" in markdown and "| 1.00 |" in markdown
    # A person confirms 1 of the 2 passes and 1 of the 2 fails.
    assert "| correctness | 1 of 2 (50.0%) | 1 of 2 (50.0%) |" in markdown
    assert "Missing: fact x." in markdown and "misses a fact" in markdown


def test_the_markdown_of_an_empty_labeling_says_so():
    markdown = render_agreement_markdown(measure_agreement(_judged_run(), []))

    assert "| correctness | 0 | — | — | — | — |" in markdown
    assert "_No verdict has been ruled on yet._" in markdown


# ------------------------------------------------------------------ the files


def test_labels_round_trip_and_are_never_overwritten(tmp_path):
    path = tmp_path / "out" / "labels.jsonl"
    rows = [_label(1, "correctness", None), _label(2, "hallucination", True, context=["c"])]

    write_labels(path, rows)
    assert read_labels(path) == rows
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    # A file that may hold hours of rulings stays as it is.
    with pytest.raises(FileExistsError, match="already exists"):
        write_labels(path, [])
    assert read_labels(path) == rows


def test_a_malformed_label_line_names_its_file_and_line(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(_label(1, "correctness", True).model_dump_json() + "\n{not json\n")

    with pytest.raises(ValueError, match=r"labels\.jsonl, line 2"):
        read_labels(path)


def test_blank_lines_in_the_labels_are_skipped(tmp_path):
    path = tmp_path / "labels.jsonl"
    line = _label(1, "correctness", True).model_dump_json()
    path.write_text(f"\n{line}\n\n   \n{line}\n", encoding="utf-8")
    assert len(read_labels(path)) == 2


# ------------------------------------------------------------------ the runner


def _prepare(tmp_path: Path, monkeypatch) -> tuple[Path, Path, EvalReport]:
    report, items, contexts = _run()
    report_path = tmp_path / "report.json"
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    dataset = tmp_path / "items.jsonl"
    dataset.write_text("\n".join(i.model_dump_json() for i in items), encoding="utf-8")

    async def load_contexts(ids: set[int]) -> dict[int, str]:
        return {i: contexts[i] for i in ids if i in contexts}

    monkeypatch.setattr(runner, "load_contexts", load_contexts)
    return report_path, dataset, report


async def test_label_then_agreement_end_to_end(tmp_path, monkeypatch):
    report_path, dataset, report = _prepare(tmp_path, monkeypatch)

    rows = await runner.run_label(report_path, dataset, 10, 0)
    labels_path = tmp_path / f"labels_{RUN_ID[:8]}.jsonl"
    assert len(rows) == 10 and labels_path.is_file()

    # Run again: the labels are not overwritten.
    with pytest.raises(FileExistsError):
        await runner.run_label(report_path, dataset, 10, 0)

    # A person rules the same as the judge on everything but two correctness verdicts.
    ruled = []
    flipped = 0
    for row in read_labels(labels_path):
        by_item = {r.item_id: r for r in report.results}
        judge = CHECKS[row.check](by_item[row.item_id]).passed
        if row.check == "correctness" and flipped < 2:
            judge, flipped = not judge, flipped + 1
        ruled.append(row.model_copy(update={"human": judge}))
    labels_path.write_text("".join(r.model_dump_json() + "\n" for r in ruled), encoding="utf-8")

    summary = runner.run_agreement(report_path, labels_path)

    correctness, grounding = summary.checks
    assert (correctness.labeled, correctness.agree) == (5, 3)
    assert (grounding.labeled, grounding.agree, grounding.kappa) == (5, 5, 1.0)
    assert len(summary.disagreements) == 2
    written = tmp_path / f"agreement_{RUN_ID[:8]}.md"
    assert "# Judge vs Human" in written.read_text(encoding="utf-8")
    assert json.loads(written.with_suffix(".json").read_text(encoding="utf-8"))["run_id"] == RUN_ID


def test_the_cli_exits_2_when_the_labels_would_overwrite_a_file(tmp_path, monkeypatch):
    report_path, dataset, _ = _prepare(tmp_path, monkeypatch)
    out = tmp_path / "mine.jsonl"
    out.write_text("someone's rulings", encoding="utf-8")
    argv = ["runner", "label", str(report_path), "--dataset", str(dataset), "-o", str(out)]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as caught:
        runner.main()

    assert caught.value.code == runner.EXIT_ABORTED
    assert out.read_text(encoding="utf-8") == "someone's rulings"


def test_the_cli_exits_2_on_a_dataset_the_run_did_not_read(tmp_path, monkeypatch):
    report_path, dataset, _ = _prepare(tmp_path, monkeypatch)
    dataset.write_text(_item(1, question="another question").model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["runner", "label", str(report_path), "--dataset", str(dataset)]
    )

    with pytest.raises(SystemExit) as caught:
        runner.main()

    assert caught.value.code == runner.EXIT_ABORTED
    assert not list(tmp_path.glob("labels_*"))


def test_the_cli_exits_2_on_a_malformed_labels_file(tmp_path, monkeypatch):
    report_path, _, _ = _prepare(tmp_path, monkeypatch)
    labels = tmp_path / "labels.jsonl"
    labels.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["runner", "agreement", str(report_path), str(labels)])

    with pytest.raises(SystemExit) as caught:
        runner.main()

    assert caught.value.code == runner.EXIT_ABORTED
