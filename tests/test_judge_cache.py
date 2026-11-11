"""The judge's cache and the eval's own temperature (audit E7): the same
answer to the same question gets the verdict it got before, with no call,
and an eval run answers at AGENT_EVAL_RAG_TEMPERATURE, not the chat's."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.api.v1 import eval as eval_api
from app.config import settings
from app.eval import judges as judges_module
from app.eval import runner
from app.eval import schemas as eval_schemas
from app.eval.judge_cache import JudgeCache, judge_cache_from_settings
from app.eval.judges import Judge
from app.eval.metrics import compute_metrics
from app.eval.schemas import Category, Difficulty, EvalReport, JudgeResult
from tests.judge_fakes import verdict_for

OLD_CLAUDE = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
# Sampled and reasoned like OLD_CLAUDE: only the model differs.
OLD_HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class _Provider:
    """Counts the judge's calls; fails them while `down`."""

    def __init__(self, passed: bool = True):
        self.calls = 0
        self.passed = passed
        self.down = False

    async def generate_structured(self, *, response_model, **_kwargs):
        self.calls += 1
        if self.down:
            raise ConnectionError("judge unreachable")
        return verdict_for(response_model, self.passed, f"call {self.calls}"), None


def _item(question: str = "q", **overrides: object) -> eval_schemas.TestItem:
    fields: dict = {
        "id": 1,
        "question": question,
        "expected_answer": "a",
        "difficulty": Difficulty.EASY,
        "category": Category.IN_SCOPE,
    }
    return eval_schemas.TestItem(**{**fields, **overrides})


async def test_the_same_answer_is_judged_once_across_runs(tmp_path):
    path = tmp_path / "cache" / "judge.jsonl"
    provider = _Provider()
    first = Judge(provider, OLD_CLAUDE, cache=JudgeCache(path))  # type: ignore[arg-type]

    verdict = await first.judge_correctness(_item(), "answer")
    grounded = await first.judge_hallucination("answer", ["context"])

    # Another run reads the file: both verdicts come from it, as they were.
    provider.passed = False
    second = Judge(provider, OLD_CLAUDE, cache=JudgeCache(path))  # type: ignore[arg-type]
    assert await second.judge_correctness(_item(), "answer") == verdict
    assert await second.judge_hallucination("answer", ["context"]) == grounded
    assert (provider.calls, first.cache_hits, second.cache_hits) == (2, 0, 2)


async def test_anything_the_verdict_depends_on_misses_the_cache(tmp_path, monkeypatch):
    path = tmp_path / "judge.jsonl"
    provider = _Provider()
    judge = Judge(provider, OLD_CLAUDE, cache=JudgeCache(path))  # type: ignore[arg-type]
    await judge.judge_correctness(_item(), "answer")

    await judge.judge_correctness(_item(), "another answer")
    await judge.judge_correctness(_item("another question"), "answer")
    await judge.judge_correctness(_item(expected_answer="b"), "answer")
    await judge.judge_hallucination("answer", ["other context"])
    other_model = Judge(provider, OLD_HAIKU, cache=JudgeCache(path))  # type: ignore[arg-type]
    await other_model.judge_correctness(_item(), "answer")
    # A model that takes a temperature: another one is another verdict.
    warmer = Judge(provider, OLD_CLAUDE, 0.5, cache=JudgeCache(path))  # type: ignore[arg-type]
    await warmer.judge_correctness(_item(), "answer")
    monkeypatch.setattr(judges_module, "JUDGE_PROMPT_VERSION", "000000000000")
    await Judge(provider, OLD_CLAUDE, cache=JudgeCache(path)).judge_correctness(_item(), "answer")  # type: ignore[arg-type]

    assert provider.calls == 8
    assert judge.cache_hits == other_model.cache_hits == warmer.cache_hits == 0


async def test_a_model_with_fixed_sampling_shares_its_verdicts_across_temperatures(tmp_path):
    # Gemini 3+ is sent no temperature: the setting does not change the call.
    provider = _Provider()
    path = tmp_path / "judge.jsonl"
    await Judge(provider, "gemini-3.8-flash", 0.0, cache=JudgeCache(path)).judge_correctness(  # type: ignore[arg-type]
        _item(), "answer"
    )
    warmer = Judge(provider, "gemini-3.8-flash", 0.7, cache=JudgeCache(path))  # type: ignore[arg-type]
    await warmer.judge_correctness(_item(), "answer")
    assert (provider.calls, warmer.cache_hits) == (1, 1)


async def test_another_thinking_level_is_another_verdict(tmp_path, monkeypatch):
    provider = _Provider()
    path = tmp_path / "judge.jsonl"
    monkeypatch.setattr(settings, "gemini_thinking_level", "low")
    await Judge(provider, "gemini-3.8-flash", cache=JudgeCache(path)).judge_correctness(  # type: ignore[arg-type]
        _item(), "answer"
    )
    monkeypatch.setattr(settings, "gemini_thinking_level", "high")
    deeper = Judge(provider, "gemini-3.8-flash", cache=JudgeCache(path))  # type: ignore[arg-type]
    await deeper.judge_correctness(_item(), "answer")
    assert (provider.calls, deeper.cache_hits) == (2, 0)


async def test_a_failed_call_is_not_kept(tmp_path):
    path = tmp_path / "judge.jsonl"
    provider = _Provider()
    provider.down = True
    judge = Judge(provider, OLD_CLAUDE, cache=JudgeCache(path))  # type: ignore[arg-type]

    failed = await judge.judge_correctness(_item(), "answer")
    provider.down = False
    retried = await judge.judge_correctness(_item(), "answer")

    assert failed.error and not retried.error
    assert provider.calls == 2
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    cache = JudgeCache(tmp_path / "other.jsonl")
    cache.put("k", JudgeResult(passed=False, reasoning="Judge call failed", error=True))
    assert cache.get("k") is None and not cache.path.exists()


async def test_a_verdict_with_no_call_does_not_touch_the_cache(tmp_path):
    path = tmp_path / "judge.jsonl"
    provider = _Provider()
    judge = Judge(provider, OLD_CLAUDE, cache=JudgeCache(path))  # type: ignore[arg-type]

    await judge.judge_correctness(_item(forbidden=["PWNED"]), "pwned")
    await judge.judge_hallucination("answer", [])

    assert provider.calls == judge.cache_hits == 0
    assert not path.exists()


async def test_a_line_that_does_not_parse_is_skipped(tmp_path):
    path = tmp_path / "judge.jsonl"
    provider = _Provider()
    await Judge(provider, OLD_CLAUDE, cache=JudgeCache(path)).judge_correctness(_item(), "answer")  # type: ignore[arg-type]
    good = path.read_text(encoding="utf-8")
    path.write_text('{"key": "cut sh\n[1, 2]\n{"passed": true}\n' + good, encoding="utf-8")

    judge = Judge(provider, OLD_CLAUDE, cache=JudgeCache(path))  # type: ignore[arg-type]
    await judge.judge_correctness(_item(), "answer")

    assert (provider.calls, judge.cache_hits) == (1, 1)


async def test_without_a_cache_every_answer_is_judged(monkeypatch):
    provider = _Provider()
    judge = Judge(provider, OLD_CLAUDE)  # type: ignore[arg-type]
    await judge.judge_correctness(_item(), "answer")
    await judge.judge_correctness(_item(), "answer")
    assert (provider.calls, judge.cache_hits) == (2, 0)

    monkeypatch.setattr(settings.agent, "eval_judge_cache", " ")
    assert judge_cache_from_settings() is None
    monkeypatch.setattr(settings.agent, "eval_judge_cache", "eval/cache/judge.jsonl")
    cache = judge_cache_from_settings()
    assert cache is not None and cache.path == Path("eval/cache/judge.jsonl")


# --------------------------------------------- where the eval builds a run


class _Captured:
    """Stands in for RAGEvaluator: keeps what it was built with."""

    built: list[dict] = []

    def __init__(self, **kwargs):
        _Captured.built.append(kwargs)

    async def run(self, dataset):
        result = eval_schemas.EvalResult(
            item_id=1,
            question="q",
            expected_answer="a",
            system_answer="a",
            citations=[],
            latency_ms=1,
            tokens_input=1,
            tokens_output=1,
            correctness=JudgeResult(passed=True, reasoning="ok"),
            hallucination=JudgeResult(passed=True, reasoning="ok"),
            retrieval_precision=1.0,
            difficulty=Difficulty.EASY,
            category=Category.IN_SCOPE,
        )
        return EvalReport(
            run_id="r" * 36,
            timestamp=datetime(2026, 11, 9, tzinfo=UTC),
            dataset_size=len(dataset),
            model="m",
            metrics=compute_metrics([result], "m"),
            results=[result],
        )


@pytest.fixture
def captured(monkeypatch, tmp_path):
    _Captured.built = []
    monkeypatch.setattr(settings.agent, "temp_rag_generation", 0.3)
    monkeypatch.setattr(settings.agent, "eval_rag_temperature", 0.1)
    monkeypatch.setattr(settings.agent, "eval_judge_cache", str(tmp_path / "judge.jsonl"))
    for module in (runner, eval_api):
        monkeypatch.setattr(module, "RAGEvaluator", _Captured)
        monkeypatch.setattr(module, "llm_service_for", lambda node: (None, f"{node}-model", 0.3))
    return _Captured.built


def _run_cli(monkeypatch, tmp_path, *flags: str) -> None:
    dataset = tmp_path / "items.jsonl"
    dataset.write_text(_item().model_dump_json() + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "EmbeddingService", lambda: None)
    argv = ["runner", "run", "--dataset", str(dataset), "--collection-id", "1"]
    argv += ["--results-dir", str(tmp_path / "results"), *flags]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()


def test_the_cli_answers_at_the_eval_temperature_with_the_cache(captured, monkeypatch, tmp_path):
    _run_cli(monkeypatch, tmp_path)
    _run_cli(monkeypatch, tmp_path, "--no-judge-cache")

    with_cache, without = captured
    assert with_cache["temperature"] == without["temperature"] == 0.1
    assert with_cache["judge"].cache.path == tmp_path / "judge.jsonl"
    assert without["judge"].cache is None


async def test_the_api_answers_at_the_eval_temperature_with_the_cache(
    captured, monkeypatch, tmp_path
):
    monkeypatch.setattr(eval_api, "RESULTS_DIR", tmp_path / "results")
    payload = eval_api.EvalRunRequest(dataset=[_item()])

    await eval_api.run_evaluation(payload, embedder=None)  # type: ignore[arg-type]

    (built,) = captured
    assert built["temperature"] == 0.1
    assert built["judge"].cache.path == tmp_path / "judge.jsonl"
