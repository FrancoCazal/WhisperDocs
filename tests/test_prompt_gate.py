"""The prompt gate (audit M-4): `runner diff` fails when the current run's
accuracy drops, or its hallucination rate rises, past a threshold over the
previous run's on the same items. Each report carries the prompt versions it
ran with."""

from __future__ import annotations

import sys
from datetime import UTC, datetime


from app.agent.prompt_version import PROMPT_VERSION
from app.config import settings
from app.eval import evaluator as evaluator_module
from app.eval import runner
from app.eval import schemas as eval_schemas
from app.eval.evaluator import RAGEvaluator
from app.eval.judges import JUDGE_PROMPT_VERSION
from app.eval.protocol import dataset_hash
from app.eval.report import diff_failures, render_diff_markdown, render_markdown
from app.eval.schemas import (
    AggregateMetrics,
    Category,
    Difficulty,
    EvalReport,
    EvalResult,
    JudgeResult,
)


def _result(item_id: int, *, correct: bool = True, grounded: bool = True) -> EvalResult:
    return EvalResult(
        item_id=item_id,
        question=f"q{item_id}",
        expected_answer="e",
        system_answer="a",
        citations=[],
        latency_ms=100,
        tokens_input=10,
        tokens_output=5,
        correctness=JudgeResult(passed=correct, reasoning="r"),
        hallucination=JudgeResult(passed=grounded, reasoning="r"),
        retrieval_precision=1.0,
        difficulty=Difficulty.EASY,
        category=Category.IN_SCOPE,
    )


def _report(
    results: list[EvalResult],
    *,
    prompt_version: str | None = "aaaaaaaaaaaa",
    judge: str | None = "jjjjjjjjjjjj",
) -> EvalReport:
    n = len(results)
    return EvalReport(
        run_id="r" * 8,
        timestamp=datetime.now(UTC),
        dataset_size=n,
        model="m",
        prompt_version=prompt_version,
        judge_prompt_version=judge,
        metrics=AggregateMetrics(
            total=n,
            accuracy=sum(r.correctness.passed for r in results) / n,
            hallucination_rate=sum(not r.hallucination.passed for r in results) / n,
            avg_retrieval_precision=1.0,
            avg_latency_ms=100.0,
            avg_tokens_input=10.0,
            avg_tokens_output=5.0,
            total_tokens_input=10 * n,
            total_tokens_output=5 * n,
            estimated_cost_usd=0.0,
            by_difficulty={},
            by_category={},
        ),
        results=results,
    )


def _items(n: int, *, wrong: int = 0, hallucinated: int = 0) -> list[EvalResult]:
    return [_result(i, correct=i >= wrong, grounded=i >= hallucinated) for i in range(n)]


def test_a_drop_up_to_the_threshold_passes():
    # 80% to 70% is ten points, a hair more in floating point.
    base, head = _report(_items(10, wrong=2)), _report(_items(10, wrong=3))

    assert diff_failures(base, head, max_accuracy_drop=0.10, max_hallucination_rise=0.10) == []


def test_a_drop_or_a_rise_past_its_threshold_fails_with_both_named():
    base, head = _report(_items(20)), _report(_items(20, wrong=3, hallucinated=3))

    failures = diff_failures(base, head, max_accuracy_drop=0.10, max_hallucination_rise=0.10)

    assert failures == [
        "accuracy dropped 15.0 points, from 100.0% to 85.0%, more than 10.0 "
        "(3 items worse, 0 better, McNemar p = 0.250)",
        "hallucination rate rose 15.0 points, from 0.0% to 15.0%, more than 10.0 "
        "(3 items worse, 0 better, McNemar p = 0.250)",
    ]


def test_an_improvement_passes_and_one_turns_the_thresholds_off():
    worse, better = _report(_items(20, wrong=10, hallucinated=10)), _report(_items(20))

    assert diff_failures(worse, better, max_accuracy_drop=0.0, max_hallucination_rise=0.0) == []
    assert diff_failures(better, worse, max_accuracy_drop=1.0, max_hallucination_rise=1.0) == []


def test_runs_over_different_items_are_not_compared():
    base = _report(_items(20))
    more = _report(_items(21))
    reworded = _report([*_items(19), _result(19).model_copy(update={"question": "other"})])

    for head in (more, reworded):
        (failure,) = diff_failures(base, head, max_accuracy_drop=1.0, max_hallucination_rise=1.0)
        assert failure.startswith("the runs answered different items")


def test_the_reports_show_their_prompt_versions_and_the_diff_warns_on_a_new_judge():
    base = _report(_items(4), prompt_version="aaaaaaaaaaaa")
    head = _report(_items(4), prompt_version="bbbbbbbbbbbb", judge="kkkkkkkkkkkk")
    old = _report(_items(4), prompt_version=None, judge=None)

    md = render_diff_markdown(base, head)

    assert "prompts `aaaaaaaaaaaa`" in md and "prompts `bbbbbbbbbbbb`" in md
    assert "judge's prompts changed (`jjjjjjjjjjjj` → `kkkkkkkkkkkk`)" in md
    assert "prompts —" in render_diff_markdown(old, head)
    assert "judge's prompts changed" not in render_diff_markdown(base, base)
    assert "judge's prompts changed" not in render_diff_markdown(old, head)
    assert "- **Prompts**: `bbbbbbbbbbbb` (judge `kkkkkkkkkkkk`)" in render_markdown(head)
    assert "- **Prompts**: — (judge —)" in render_markdown(old)


def _diff(monkeypatch, tmp_path, capsys, base, head, *flags: str) -> tuple[int, str]:
    paths = []
    for name, report in (("base", base), ("head", head)):
        path = tmp_path / f"{name}.json"
        path.write_text(report.model_dump_json(), encoding="utf-8")
        paths.append(str(path))
    output = tmp_path / "diff.md"
    monkeypatch.setattr(sys, "argv", ["runner", "diff", *paths, "-o", str(output), *flags])
    code = 0
    try:
        runner.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    assert output.read_text(encoding="utf-8").startswith("# Eval Diff")
    return code, capsys.readouterr().err


def test_the_cli_exits_3_past_the_configured_threshold_and_still_writes_the_diff(
    monkeypatch, tmp_path, capsys
):
    base, head = _report(_items(20)), _report(_items(20, wrong=3, hallucinated=3))
    monkeypatch.setattr(settings.agent, "eval_max_accuracy_drop", 0.10)
    monkeypatch.setattr(settings.agent, "eval_max_hallucination_rise", 0.20)

    failing, err = _diff(monkeypatch, tmp_path, capsys, base, head)
    monkeypatch.setattr(settings.agent, "eval_max_accuracy_drop", 0.20)
    passing, quiet = _diff(monkeypatch, tmp_path, capsys, base, head)

    assert failing == runner.EXIT_BELOW_THRESHOLD
    assert "accuracy dropped 15.0 points" in err
    assert "hallucination" not in err
    assert passing == 0
    assert quiet == ""


def test_the_flags_override_the_settings(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(settings.agent, "eval_max_accuracy_drop", 0.10)
    monkeypatch.setattr(settings.agent, "eval_max_hallucination_rise", 0.10)
    base = _report(_items(20))
    less_accurate = _report(_items(20, wrong=3))
    one_hallucination = _report(_items(20, hallucinated=1))
    diff = (monkeypatch, tmp_path, capsys, base)

    lenient, _ = _diff(*diff, less_accurate, "--max-accuracy-drop", "0.2")
    strict, err = _diff(*diff, one_hallucination, "--max-hallucination-rise", "0")

    assert lenient == 0
    assert strict == runner.EXIT_BELOW_THRESHOLD
    assert "hallucination rate rose 5.0 points" in err


class _Judge:
    model = "fake-judge"
    temperature = 0.0
    cache = None
    cache_hits = 0

    async def judge_correctness(self, _item, _answer) -> JudgeResult:
        return JudgeResult(passed=True, reasoning="ok")

    async def judge_hallucination(self, _answer, _contexts) -> JudgeResult:
        return JudgeResult(passed=True, reasoning="ok")


class _NoSession:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *_exc):
        return False


class _Embedder:
    async def embed(self, _text: str) -> list[float]:
        return [0.0]


async def test_a_run_reports_the_prompt_versions_it_answered_and_judged_with(monkeypatch):
    async def nothing_retrieved(**_kwargs):
        return []

    async def fingerprint(_db, collection_id):
        assert collection_id == 1
        return 95, "c" * 12

    monkeypatch.setattr(evaluator_module, "retrieve_pipeline", nothing_retrieved)
    monkeypatch.setattr(evaluator_module, "collection_fingerprint", fingerprint)
    evaluator = RAGEvaluator(
        session_maker=_NoSession,  # type: ignore[arg-type]
        embedder=_Embedder(),  # type: ignore[arg-type]
        provider=None,  # type: ignore[arg-type]
        model="fake-model",
        temperature=0.0,
        judge=_Judge(),  # type: ignore[arg-type]
        collection_id=1,
    )
    item = eval_schemas.TestItem(
        id=1,
        question="q",
        expected_answer="a",
        difficulty=Difficulty.EASY,
        category=Category.OUT_OF_SCOPE,
    )

    report = await evaluator.run([item])

    assert report.prompt_version == PROMPT_VERSION
    assert report.judge_prompt_version == JUDGE_PROMPT_VERSION
    # And the settings it ran with (audit E7).
    assert report.config is not None
    assert report.config.dataset_hash == dataset_hash([item])
    assert (report.config.collection_id, report.config.top_k) == (1, evaluator.top_k)
    # The chunks its collection held as it started.
    assert (report.config.collection_chunks, report.config.collection_hash) == (95, "c" * 12)
    assert report.config.concurrency == evaluator.concurrency
    assert report.config.judge_model == "fake-judge"
    assert report.config.rag_temperature == 0.0
    assert report.config.judge_cache is False and report.judge_cache_hits is None
