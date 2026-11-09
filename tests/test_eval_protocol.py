"""What it takes to compare two eval runs (audit E7, E8): each report says
what it ran with, a share comes with its interval, and a diff says which
items flipped, whether the flips lean one way, and when two runs are not
comparable at all."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config import settings
from app.eval import protocol
from app.eval.protocol import (
    dataset_hash,
    git_revision,
    mcnemar_exact,
    run_config,
    sampling,
    wilson_interval,
)
from app.eval import schemas as eval_schemas
from app.eval.report import (
    diff_failures,
    item_changes,
    render_diff_markdown,
    render_markdown,
)
from app.eval.schemas import (
    AggregateMetrics,
    Category,
    Difficulty,
    EvalReport,
    EvalResult,
    JudgeResult,
    RunConfig,
)

OLD_CLAUDE = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
NEW_GEMINI = "gemini-3.8-flash"


def _item(item_id: int, question: str = "q", expected: str = "a") -> eval_schemas.TestItem:
    return eval_schemas.TestItem(
        id=item_id,
        question=question,
        expected_answer=expected,
        difficulty=Difficulty.EASY,
        category=Category.IN_SCOPE,
        expected_sources=["doc.md"],
    )


# ---------------------------------------------------------------- statistics


def test_the_wilson_interval_of_48_out_of_50():
    low, high = wilson_interval(48, 50)
    assert (round(low, 3), round(high, 3)) == (0.865, 0.989)


def test_the_wilson_interval_holds_at_the_edges():
    assert wilson_interval(50, 50)[1] == pytest.approx(1.0)
    assert wilson_interval(0, 50)[0] == pytest.approx(0.0)
    assert 0.9 < wilson_interval(50, 50)[0] < 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("worse", "better", "p"),
    [
        (0, 0, 1.0),
        # Six flips one way and none back is the least that reads as a change.
        (6, 0, 0.03125),
        (0, 6, 0.03125),
        (5, 0, 0.0625),
        (3, 3, 1.0),
        (1, 6, 0.125),
        (0, 30, 2 / 2**30),
    ],
)
def test_mcnemar_exact(worse, better, p):
    assert mcnemar_exact(worse, better) == pytest.approx(p)


# ------------------------------------------------------------------- config


def test_the_dataset_hash_follows_what_the_run_read():
    items = [_item(1), _item(2)]
    assert dataset_hash(items) == dataset_hash([_item(1), _item(2)])
    assert len(dataset_hash(items)) == 12
    # Another expected answer, another order, a subset: another dataset.
    assert dataset_hash([_item(1), _item(2, expected="b")]) != dataset_hash(items)
    assert dataset_hash([_item(2), _item(1)]) != dataset_hash(items)
    assert dataset_hash(items[:1]) != dataset_hash(items)


def test_git_sha_names_the_commit(monkeypatch):
    monkeypatch.setenv("GIT_SHA", "def456")
    assert git_revision() == ("def456", None)


def test_the_checkout_beats_the_workflow_commit(monkeypatch, tmp_path):
    # The prompt gate runs the base from a second checkout, while GITHUB_SHA
    # names the head: the base's report must name the base.
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setenv("GITHUB_SHA", "head000")
    monkeypatch.setattr(protocol, "_git", lambda *args: "base000" if args[0] == "rev-parse" else "")
    assert git_revision() == ("base000", False)

    # With no git and no .git, the workflow's commit is all there is.
    def no_git(*_args: str) -> str:
        raise FileNotFoundError("git")

    monkeypatch.setattr(protocol, "_git", no_git)
    monkeypatch.setattr(protocol, "REPO_ROOT", tmp_path)
    assert git_revision() == ("head000", None)


def test_without_git_the_commit_is_read_from_the_repository(monkeypatch, tmp_path):
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    def no_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    monkeypatch.setattr(protocol, "REPO_ROOT", tmp_path)

    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs" / "heads" / "main").write_text("1111111\n", encoding="utf-8")
    assert git_revision() == ("1111111", None)

    # A branch only in packed-refs.
    (git / "HEAD").write_text("ref: refs/heads/packed\n", encoding="utf-8")
    (git / "packed-refs").write_text(
        "# pack-refs with: peeled\n2222222 refs/heads/packed\n", encoding="utf-8"
    )
    assert git_revision() == ("2222222", None)

    # A detached HEAD holds the commit itself.
    (git / "HEAD").write_text("3333333\n", encoding="utf-8")
    assert git_revision() == ("3333333", None)

    monkeypatch.setattr(protocol, "REPO_ROOT", tmp_path / "nowhere")
    assert git_revision() == (None, None)


def test_this_checkout_names_its_commit(monkeypatch):
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    if not (Path(protocol.REPO_ROOT) / ".git").exists():
        pytest.skip("not a git checkout")
    sha, _dirty = git_revision()
    assert sha is not None and len(sha) == 40


def test_a_model_with_fixed_sampling_records_no_temperature():
    assert sampling(OLD_CLAUDE, 0.0) == 0.0
    assert sampling(NEW_GEMINI, 0.0) is None
    assert sampling("us.anthropic.claude-sonnet-5-5", 0.3) is None


def test_the_config_records_the_settings_that_change_a_run(monkeypatch):
    monkeypatch.setattr(settings, "reranker_enabled", False)
    monkeypatch.setattr(settings, "mmr_enabled", True)
    monkeypatch.setattr(settings, "mmr_lambda", 0.7)
    monkeypatch.setenv("GIT_SHA", "abc123")
    items = [_item(1)]

    config = run_config(
        items=items,
        collection_id=3,
        top_k=5,
        concurrency=4,
        rag_model=OLD_CLAUDE,
        rag_temperature=0.0,
        judge_model=NEW_GEMINI,
        judge_temperature=0.0,
        judge_cache=True,
    )

    assert config.git_sha == "abc123"
    assert config.dataset_hash == dataset_hash(items)
    assert (config.collection_id, config.top_k, config.concurrency) == (3, 5, 4)
    assert config.rag_temperature == 0.0
    assert config.judge_temperature is None
    assert config.judge_reasoning == settings.gemini_thinking_level
    assert config.judge_cache is True
    assert config.reranker_model is None
    assert config.mmr_lambda == 0.7
    assert config.embedding_model == settings.embedding_model


# ------------------------------------------------------------------ reports


def _verdict(passed: bool, error: bool = False) -> JudgeResult:
    return JudgeResult(passed=passed, reasoning="ok" if passed else "wrong", error=error)


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
        correctness=_verdict(correct, error),
        hallucination=_verdict(grounded),
        retrieval_precision=1.0,
        difficulty=Difficulty.EASY,
        category=Category.IN_SCOPE,
    )


def _config(**overrides: object) -> RunConfig:
    fields: dict = {
        "git_sha": "a" * 40,
        "git_dirty": False,
        "dataset_hash": "d" * 12,
        "collection_id": 1,
        "top_k": 5,
        "concurrency": 4,
        "rag_temperature": 0.0,
        "rag_reasoning": None,
        "rag_structured_output": True,
        "rag_context_token_budget": 6000,
        "judge_model": NEW_GEMINI,
        "judge_temperature": None,
        "judge_reasoning": "low",
        "judge_cache": True,
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 1536,
        "reranker_model": None,
        "reranker_candidate_pool": 20,
        "reranker_score_threshold": 0.0,
        "mmr_lambda": None,
    }
    return RunConfig(**{**fields, **overrides})


def _report(results: list[EvalResult], config: RunConfig | None = None) -> EvalReport:
    total = len(results)
    correct = sum(r.correctness.passed for r in results)
    hallucinated = sum(not r.hallucination.passed for r in results)
    return EvalReport(
        run_id="r" * 8,
        timestamp=datetime(2026, 11, 9, tzinfo=UTC),
        dataset_size=total,
        model=OLD_CLAUDE,
        prompt_version="p" * 12,
        judge_prompt_version="j" * 12,
        config=config,
        metrics=AggregateMetrics(
            total=total,
            accuracy=correct / total,
            hallucination_rate=hallucinated / total,
            avg_retrieval_precision=1.0,
            avg_latency_ms=1000.0,
            avg_tokens_input=100.0,
            avg_tokens_output=10.0,
            total_tokens_input=100 * total,
            total_tokens_output=10 * total,
            estimated_cost_usd=0.01,
            by_difficulty={},
            by_category={},
        ),
        results=results,
    )


def _row(markdown: str, label: str) -> str:
    return next(line for line in markdown.splitlines() if line.startswith(f"| {label} |"))


def test_a_report_shows_its_intervals_and_its_config():
    report = _report([_result(i, i <= 48) for i in range(1, 51)], _config())
    md = render_markdown(report)
    assert _row(md, "Accuracy, 95% interval") == "| Accuracy, 95% interval | 86.5% – 98.9% |"
    assert _row(md, "dataset_hash") == "| dataset_hash | `dddddddddddd` |"
    # A setting that does not apply shows as missing, not as "None".
    assert _row(md, "judge_temperature") == "| judge_temperature | — |"


def test_a_report_says_how_many_verdicts_came_from_the_cache():
    report = _report([_result(1, True)], _config())
    assert "from the cache" not in render_markdown(report)
    cached = report.model_copy(update={"judge_cache_hits": 3})
    assert "- **Judge verdicts from the cache**: 3" in render_markdown(cached)


def test_an_old_report_says_it_has_no_config():
    md = render_markdown(_report([_result(1, True)]))
    assert "_This report is older than the run config._" in md


def test_the_diff_lists_regressions_fixes_and_their_mcnemar_p():
    before = [_result(i, correct=i > 6) for i in range(1, 21)]
    # Items 1-6 now pass, item 20 now fails; item 7's judge call failed,
    # which fails it by construction.
    after = [_result(i, correct=i not in (7, 20), error=i == 7) for i in range(1, 21)]
    md = render_diff_markdown(_report(before, _config()), _report(after, _config()))

    assert _row(md, "correctness") == "| correctness | 1 | 6 | 0.125 |"
    assert _row(md, "hallucination") == "| hallucination | 0 | 0 | 1.000 |"
    regressions = md.split("### Regressions")[1].split("### Fixes")[0]
    fixes = md.split("### Fixes")[1]
    assert "| 20 | correctness | question 20 |" in regressions
    assert [f"| {i} | correctness |" in fixes for i in range(1, 7)] == [True] * 6
    # A failed judge call is no flip.
    assert "| 7 |" not in md
    assert "_Nothing recorded differs._" in md


def test_a_reworded_question_is_no_flip():
    before = [_result(1, correct=True), _result(2, correct=True)]
    after = [
        _result(1, correct=False).model_copy(update={"question": "other"}),
        _result(2, correct=False),
    ]
    changes = item_changes(_report(before), _report(after))
    assert [(c.item_id, c.check, c.regression) for c in changes] == [(2, "correctness", True)]


def test_the_diff_names_what_changed_and_warns_on_the_judge():
    prev = _report([_result(1, True)], _config())
    curr = _report([_result(1, True)], _config(top_k=8, judge_model="gemini-4-flash"))
    md = render_diff_markdown(prev, curr)

    assert _row(md, "top_k") == "| top_k | `5` | `8` |"
    assert _row(md, "judge_model") == f"| judge_model | `{NEW_GEMINI}` | `gemini-4-flash` |"
    assert "> The judge changed (judge_model)" in md
    assert "| collection_id |" not in md


def test_the_diff_warns_on_uncommitted_code_and_on_an_old_report():
    dirty = _report([_result(1, True)], _config(git_dirty=True))
    md = render_diff_markdown(_report([_result(1, True)]), dirty)
    assert "> The current run had uncommitted changes on top of `aaaaaaaaaaaa`" in md
    assert "> A report is older than the run config" in md


def test_runs_over_different_datasets_are_not_compared():
    prev = _report([_result(1, True)], _config(dataset_hash="111111111111"))
    curr = _report([_result(1, False)], _config(dataset_hash="222222222222"))

    failures = diff_failures(prev, curr, max_accuracy_drop=0.1, max_hallucination_rise=0.1)

    assert failures == [
        "the runs read different datasets (111111111111 and 222222222222): the same "
        "questions with other expected answers or sources, so they are not compared"
    ]
    assert "> The runs read different datasets" in render_diff_markdown(prev, curr)


def test_a_failing_diff_says_how_many_items_flipped_and_its_p():
    before = [_result(i, correct=True) for i in range(1, 11)]
    after = [_result(i, correct=i > 6) for i in range(1, 11)]

    failures = diff_failures(
        _report(before, _config()),
        _report(after, _config()),
        max_accuracy_drop=0.1,
        max_hallucination_rise=0.1,
    )

    assert failures == [
        "accuracy dropped 60.0 points, from 100.0% to 40.0%, more than 10.0 "
        "(6 items worse, 0 better, McNemar p = 0.031)"
    ]
