"""`GET /eval/results/{run_id}`: a saved run in full, so the UI can show
each item's answers and verdicts and compare two runs (frontend review U5).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from app.api.v1 import eval as eval_api
from app.eval.metrics import compute_metrics
from app.eval.schemas import (
    Category,
    Difficulty,
    EvalReport,
    EvalResult,
    JudgeResult,
    RunConfig,
)
from app.main import app

RUN_ID = "b2b0899a-fc90-4c4e-9a37-2f1d0d6a3b10"
# Same first 8 characters: the file name alone cannot tell them apart.
TWIN_ID = "b2b0899a-0000-4000-8000-000000000000"


def _report(run_id: str, answer: str) -> EvalReport:
    result = EvalResult(
        item_id=1,
        question="What distance metrics does pgvector support?",
        expected_answer="cosine, L2 and inner product",
        system_answer=answer,
        citations=["pgvector.md"],
        latency_ms=1200,
        tokens_input=900,
        tokens_output=80,
        correctness=JudgeResult(passed=True, reasoning="matches"),
        hallucination=JudgeResult(passed=True, reasoning="grounded"),
        retrieval_precision=1.0,
        retrieval_recall=1.0,
        difficulty=Difficulty.EASY,
        category=Category.IN_SCOPE,
    )
    return EvalReport(
        run_id=run_id,
        timestamp=datetime(2026, 11, 7, tzinfo=UTC),
        dataset_size=1,
        model="test-model",
        metrics=compute_metrics([result], "test-model"),
        results=[result],
    )


@pytest.fixture
def results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(eval_api, "RESULTS_DIR", tmp_path)
    return tmp_path


def _save(directory: Path, name: str, report: EvalReport) -> None:
    (directory / name).write_text(report.model_dump_json(), encoding="utf-8")


async def _get(path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
        return await c.get(path)


async def test_a_run_is_read_in_full(results_dir: Path):
    _save(results_dir, "2026-11-07T10-00-00_b2b0899a.json", _report(RUN_ID, "cosine, L2, IP"))
    _save(results_dir, "2026-11-07T11-00-00_b2b0899a.json", _report(TWIN_ID, "other run"))

    r = await _get(f"/eval/results/{RUN_ID}")

    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == RUN_ID
    assert body["results"][0]["system_answer"] == "cosine, L2, IP"
    assert body["results"][0]["correctness"] == {
        "passed": True,
        "reasoning": "matches",
        "error": False,
    }
    assert body["metrics"]["latency_p95_ms"] == 1200
    assert body["metrics"]["avg_retrieval_recall"] == 1.0


def _old_report_json() -> str:
    """A report saved before recall, the latency percentiles, the judge's
    error flag and the prompt versions existed, like the April runs in
    eval/results."""
    data = _report(RUN_ID, "x").model_dump(mode="json")
    del data["prompt_version"], data["judge_prompt_version"]
    for key in ("avg_retrieval_recall", "latency_p50_ms", "latency_p95_ms", "judge_errors"):
        del data["metrics"][key]
    for item in data["results"]:
        del item["retrieval_recall"]
        del item["correctness"]["error"]
        del item["hallucination"]["error"]
    return json.dumps(data)


async def test_an_older_report_leaves_out_what_it_never_measured(results_dir: Path):
    old = results_dir / "2026-04-20T18-35-50_b2b0899a.json"
    old.write_text(_old_report_json(), encoding="utf-8")

    full = (await _get(f"/eval/results/{RUN_ID}")).json()
    summary = (await _get("/eval/results")).json()[0]

    # Not a 0 that reads as "p95 was 0 ms" or "recall was 0%".
    for key in ("avg_retrieval_recall", "latency_p50_ms", "latency_p95_ms", "judge_errors"):
        assert key not in full["metrics"]
    assert "retrieval_recall" not in full["results"][0]
    assert "error" not in full["results"][0]["correctness"]
    # What it did measure is still there.
    assert full["metrics"]["avg_latency_ms"] == 1200
    assert full["results"][0]["retrieval_precision"] == 1.0
    assert summary["avg_retrieval_recall"] is None
    assert summary["latency_p95_ms"] is None
    assert summary["avg_latency_ms"] == 1200
    assert "prompt_version" not in full and "judge_prompt_version" not in full
    assert summary["prompt_version"] is None and summary["judge_prompt_version"] is None


async def test_a_run_and_its_summary_name_its_prompt_versions(results_dir: Path):
    report = _report(RUN_ID, "x").model_copy(
        update={"prompt_version": "a286c8ec04d9", "judge_prompt_version": "6bad93eae2c1"}
    )
    _save(results_dir, "2026-11-07T10-00-00_b2b0899a.json", report)

    full = (await _get(f"/eval/results/{RUN_ID}")).json()
    (summary,) = (await _get("/eval/results")).json()

    for body in (full, summary):
        assert body["prompt_version"] == "a286c8ec04d9"
        assert body["judge_prompt_version"] == "6bad93eae2c1"


@pytest.mark.parametrize("run_id", ["9c1d7e52-0000-4000-8000-000000000000", "..", "not-a-run"])
async def test_an_unknown_or_malformed_run_is_a_404(results_dir: Path, run_id: str):
    _save(results_dir, "2026-11-07T10-00-00_b2b0899a.json", _report(RUN_ID, "x"))

    r = await _get(f"/eval/results/{run_id}")

    assert r.status_code == 404


async def test_the_limit_counts_reports_not_other_files(results_dir: Path):
    # Named so they sort before every report, newest first.
    (results_dir / "latest-badge.json").write_text('{"schemaVersion": 1}', encoding="utf-8")
    (results_dir / "agent_1777200474.json").write_text("[]", encoding="utf-8")
    _save(results_dir, "2026-11-07T10-00-00_b2b0899a.json", _report(RUN_ID, "x"))

    r = await _get("/eval/results?limit=1")

    assert [s["run_id"] for s in r.json()] == [RUN_ID]


LATER_ID = "c3d4e5f6-0000-4000-8000-000000000000"


def _config(**changes) -> RunConfig:
    values = {
        "git_sha": "0123456789abcdef0123",
        "git_dirty": False,
        "dataset_hash": "aaaaaaaaaaaa",
        "collection_id": 1,
        "top_k": 5,
        "concurrency": 4,
        "rag_temperature": None,
        "rag_reasoning": "low",
        "rag_structured_output": True,
        "rag_context_token_budget": 6000,
        "judge_model": "gemini-judge",
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
    return RunConfig(**{**values, **changes})


def _two_items(run_id: str, *, first_right: bool, second_grounded: bool, **report) -> EvalReport:
    base = _report(run_id, "x")
    one = base.results[0]
    results = [
        one.model_copy(
            update={"correctness": JudgeResult(passed=first_right, reasoning=f"one {run_id[:4]}")}
        ),
        one.model_copy(
            update={
                "item_id": 2,
                "question": "Which index types does pgvector have?",
                "hallucination": JudgeResult(passed=second_grounded, reasoning="two"),
            }
        ),
    ]
    return base.model_copy(
        update={
            "dataset_size": 2,
            "metrics": compute_metrics(results, "test-model"),
            "results": results,
            **report,
        }
    )


async def test_two_runs_are_compared_as_the_cli_compares_them(results_dir: Path):
    before = _two_items(
        RUN_ID, first_right=True, second_grounded=False, config=_config(), prompt_version="aaa"
    )
    after = _two_items(
        LATER_ID,
        first_right=False,
        second_grounded=True,
        config=_config(top_k=8, judge_model="other-judge", git_dirty=True),
        prompt_version="bbb",
        timestamp=datetime(2026, 11, 8, tzinfo=UTC),
    )
    _save(results_dir, "2026-11-07T10-00-00_b2b0899a.json", before)
    _save(results_dir, "2026-11-08T10-00-00_c3d4e5f6.json", after)

    r = await _get(f"/eval/diff?previous={RUN_ID}&current={LATER_ID}")

    assert r.status_code == 200
    diff = r.json()
    assert (diff["previous"], diff["current"]) == (RUN_ID, LATER_ID)
    assert (diff["common_items"], diff["same_items"]) == (2, True)
    assert diff["changed"] == [
        {"setting": "prompt_version", "previous": "aaa", "current": "bbb"},
        {"setting": "git_dirty", "previous": False, "current": True},
        {"setting": "top_k", "previous": 5, "current": 8},
        {"setting": "judge_model", "previous": "gemini-judge", "current": "other-judge"},
    ]
    assert any(w.startswith("The judge changed (judge_model)") for w in diff["warnings"])
    assert any(w.startswith("The current run had uncommitted changes") for w in diff["warnings"])
    assert diff["checks"] == [
        {"check": "correctness", "worse": 1, "better": 0, "p_value": 1.0},
        {"check": "hallucination", "worse": 0, "better": 1, "p_value": 1.0},
    ]
    assert diff["flips"] == [
        {
            "item_id": 1,
            "check": "correctness",
            "question": "What distance metrics does pgvector support?",
            "reasoning": "one c3d4",
            "regression": True,
        },
        {
            "item_id": 2,
            "check": "hallucination",
            "question": "Which index types does pgvector have?",
            "reasoning": "two",
            "regression": False,
        },
    ]
    accuracy = next(m for m in diff["metrics"] if m["metric"] == "accuracy")
    assert (accuracy["previous"], accuracy["current"]) == (1.0, 0.5)


async def test_an_older_run_compares_as_far_as_it_was_measured(results_dir: Path):
    (results_dir / "2026-04-20T18-35-50_b2b0899a.json").write_text(
        _old_report_json(), encoding="utf-8"
    )
    later = _report(LATER_ID, "x").model_copy(
        update={"config": _config(), "timestamp": datetime(2026, 11, 8, tzinfo=UTC)}
    )
    _save(results_dir, "2026-11-08T10-00-00_c3d4e5f6.json", later)

    diff = (await _get(f"/eval/diff?previous={RUN_ID}&current={LATER_ID}")).json()

    metrics = {m["metric"]: (m["previous"], m["current"]) for m in diff["metrics"]}
    assert metrics["latency_p95_ms"] == (None, 1200)
    assert metrics["avg_latency_ms"] == (1200, 1200)
    assert diff["changed"] == []
    assert any("older than the run config" in w for w in diff["warnings"])


def _items_run(run_id: str, questions: dict[int, str], *, right: bool, day: int) -> EvalReport:
    base = _report(run_id, "x")
    results = [
        base.results[0].model_copy(
            update={
                "item_id": item_id,
                "question": question,
                "correctness": JudgeResult(passed=right, reasoning=f"{item_id} {run_id[:4]}"),
            }
        )
        for item_id, question in questions.items()
    ]
    return base.model_copy(
        update={
            "timestamp": datetime(2026, 11, day, tzinfo=UTC),
            "dataset_size": len(results),
            "metrics": compute_metrics(results, "test-model"),
            "results": results,
        }
    )


async def test_items_compare_only_with_the_same_question_and_the_flips_get_a_p(
    results_dir: Path,
):
    # Six items both runs asked; the seventh was rewritten under its id, and
    # the later run adds an eighth.
    asked = {i: f"Question {i}?" for i in range(1, 8)}
    later = {**asked, 7: "Question 7, rewritten?", 8: "Question 8?"}
    _save(
        results_dir,
        "2026-11-07T10-00-00_b2b0899a.json",
        _items_run(RUN_ID, asked, right=True, day=7),
    )
    _save(
        results_dir,
        "2026-11-08T10-00-00_c3d4e5f6.json",
        _items_run(LATER_ID, later, right=False, day=8),
    )

    diff = (await _get(f"/eval/diff?previous={RUN_ID}&current={LATER_ID}")).json()

    assert (diff["common_items"], diff["same_items"]) == (6, False)
    # Six regressions and no fix: 2 / 2**6, as the CLI's McNemar gives it.
    assert diff["checks"][0] == {
        "check": "correctness",
        "worse": 6,
        "better": 0,
        "p_value": 0.03125,
    }
    assert [f["item_id"] for f in diff["flips"]] == [1, 2, 3, 4, 5, 6]


async def test_a_diff_with_an_unknown_run_is_a_404_that_names_it(results_dir: Path):
    _save(results_dir, "2026-11-07T10-00-00_b2b0899a.json", _report(RUN_ID, "x"))

    r = await _get(f"/eval/diff?previous={RUN_ID}&current={LATER_ID}")

    assert r.status_code == 404
    assert r.json()["detail"] == "Run c3d4e5f6 not found"
