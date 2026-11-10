"""How consistent the judge is (audit E7): `runner rejudge` judges a saved
run's answers again, with no cache, and counts how often the verdicts agree.
Only verdicts the judge gave count, and only over the context it read."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.config import settings
from app.eval import runner
from app.eval import schemas as eval_schemas
from app.eval.judges import JUDGE_PROMPT_VERSION, Judge
from app.eval.metrics import compute_metrics
from app.eval.protocol import context_hash, dataset_hash
from app.eval.rejudge import DatasetMismatchError, items_for, rejudge
from app.eval.report import render_rejudge_markdown
from app.eval.schemas import Category, Difficulty, EvalReport, EvalResult, JudgeResult
from app.models import Chunk, Collection, Document

JUDGE = "gemini-3.8-flash"


class _Provider:
    """The judge's verdicts, by the answer in the prompt: "wobbly" gets
    fail, pass, pass in turn, "contrary" always fails, "down" fails the
    call, any other passes."""

    def __init__(self):
        self.calls: dict[str, int] = {}

    async def generate_structured(self, *, messages, **_kwargs):
        prompt = messages[0]["content"]
        n = self.calls[prompt] = self.calls.get(prompt, 0) + 1
        if "down" in prompt:
            raise ConnectionError("judge unreachable")
        passed = (False, True, True)[(n - 1) % 3] if "wobbly" in prompt else True
        passed = passed and "contrary" not in prompt
        return JudgeResult(passed=passed, reasoning=f"call {n}"), None


def _item(item_id: int, **overrides: object) -> eval_schemas.TestItem:
    fields: dict = {
        "id": item_id,
        "question": f"question {item_id}",
        "expected_answer": "a",
        "difficulty": Difficulty.EASY,
        "category": Category.IN_SCOPE,
    }
    return eval_schemas.TestItem(**{**fields, **overrides})


def _verdict(passed: bool, error: bool = False) -> JudgeResult:
    return JudgeResult(passed=passed, reasoning="run", error=error)


def _result(
    item_id: int,
    answer: str,
    *,
    correct: JudgeResult | None = None,
    ids: list[int] | None = None,
    texts: list[str] | None = None,
) -> EvalResult:
    return EvalResult(
        item_id=item_id,
        question=f"question {item_id}",
        expected_answer="a",
        system_answer=answer,
        citations=[],
        latency_ms=1,
        tokens_input=1,
        tokens_output=1,
        correctness=correct or _verdict(True),
        hallucination=_verdict(True),
        retrieval_precision=1.0,
        context_chunk_ids=ids,
        context_hash=None if texts is None else context_hash(texts),
        difficulty=Difficulty.EASY,
        category=Category.IN_SCOPE,
    )


CONTEXTS = {10: "the context", 11: "rewritten since the run"}


def _run() -> tuple[EvalReport, list[eval_schemas.TestItem]]:
    results = [
        # Steady on both checks, over a context still there.
        _result(1, "steady", ids=[10], texts=["the context"]),
        # The judge rules the other way once in three; no context.
        _result(2, "wobbly", ids=[], texts=[]),
        # Did what an injection asked: fails by rule, nothing to ask again.
        _result(3, "PWNED", correct=_verdict(False), ids=[], texts=[]),
        # The run's call failed; a chunk deleted since.
        _result(4, "steady", correct=_verdict(False, error=True), ids=[99], texts=["gone"]),
        # A chunk whose text changed under the same id.
        _result(5, "steady", ids=[11], texts=["the text the judge read"]),
        # Every call fails now.
        _result(6, "down", ids=None),
        # The judge agrees with itself every time, and not with the run.
        _result(7, "contrary", ids=None),
    ]
    items = [_item(r.item_id, forbidden=["pwned"] if r.item_id == 3 else []) for r in results]
    report = EvalReport(
        run_id="abcdef12-run",
        timestamp=datetime(2026, 11, 9, tzinfo=UTC),
        dataset_size=len(results),
        model="m",
        judge_prompt_version=JUDGE_PROMPT_VERSION,
        metrics=compute_metrics(results, "m"),
        results=results,
    )
    return report, items


async def test_rejudging_counts_agreement_over_the_verdicts_the_judge_gave():
    report, items = _run()
    provider = _Provider()
    judge = Judge(provider, JUDGE)  # type: ignore[arg-type]

    summary = await rejudge(report, items, CONTEXTS, judge, times=3, concurrency=1)

    correctness, hallucination = summary.checks
    assert correctness.model_dump() == {
        "check": "correctness",
        # Item 3 is skipped, every call for item 6 failed. Item 7 agrees with
        # itself, not with the run: not unanimous.
        "items": 5,
        "unanimous": 3,
        # Item 4's run verdict was a failed call: nothing to set against.
        "verdicts_compared": 12,
        "flipped": 4,
        "judge_errors": 3,
        "skipped": 1,
    }
    assert hallucination.model_dump() == {
        "check": "hallucination",
        "items": 1,
        "unanimous": 1,
        "verdicts_compared": 3,
        "flipped": 0,
        "judge_errors": 0,
        # No context (2, 3, 6, 7), a chunk gone (4), a chunk rewritten (5).
        "skipped": 6,
    }
    assert [(d.item_id, d.check, d.original, d.again) for d in summary.disagreements] == [
        (2, "correctness", True, [False, True, True]),
        (7, "correctness", True, [False, False, False]),
    ]
    # Three calls per answer the judge rules on, none from a cache.
    assert sorted(provider.calls.values()) == [3] * 7
    assert summary.judge_prompt_version == JUDGE_PROMPT_VERSION


async def test_concurrent_calls_count_the_same():
    report, items = _run()
    judge = Judge(_Provider(), JUDGE)  # type: ignore[arg-type]
    summary = await rejudge(report, items, CONTEXTS, judge, times=3, concurrency=8)
    assert [(c.items, c.unanimous, c.flipped) for c in summary.checks] == [(5, 3, 4), (1, 1, 0)]


def test_the_items_come_from_the_dataset_the_run_read():
    report, items = _run()
    assert items_for(report, list(reversed(items))) == items

    with pytest.raises(DatasetMismatchError, match="item 6"):
        items_for(report, items[:5])
    reworded = [*items[:5], _item(6, question="another")]
    with pytest.raises(DatasetMismatchError, match="item 6"):
        items_for(report, reworded)

    config = _config(dataset_hash(items))
    assert items_for(report.model_copy(update={"config": config}), items) == items
    other = [*items[:5], _item(6, expected_answer="b"), items[6]]
    with pytest.raises(DatasetMismatchError, match="not the one the run read"):
        items_for(report.model_copy(update={"config": config}), other)


def _config(hash_: str) -> eval_schemas.RunConfig:
    return eval_schemas.RunConfig(
        git_sha=None,
        git_dirty=None,
        dataset_hash=hash_,
        collection_id=1,
        top_k=5,
        concurrency=4,
        rag_temperature=None,
        rag_reasoning=None,
        rag_structured_output=True,
        rag_context_token_budget=6000,
        judge_model="older-judge",
        judge_temperature=None,
        judge_reasoning=None,
        judge_cache=True,
        embedding_model="e",
        embedding_dimension=1536,
        reranker_model=None,
        reranker_candidate_pool=20,
        reranker_score_threshold=0.0,
        mmr_lambda=None,
    )


async def test_the_markdown_shows_agreement_and_disagreements():
    report, items = _run()
    summary = await rejudge(
        report,
        items,
        CONTEXTS,
        Judge(_Provider(), JUDGE),
        times=3,
        concurrency=1,  # type: ignore[arg-type]
    )
    md = render_rejudge_markdown(summary)

    assert "| correctness | 5 | 3 (60.0%) | 4 of 12 (33.3%) | 3 | 1 |" in md
    assert "| hallucination | 1 | 1 (100.0%) | 0 of 3 (0.0%) | 0 | 6 |" in md
    assert "| 7 | correctness | pass | fail, fail, fail | question 7 |" in md
    assert "| 2 | correctness | pass | fail, pass, pass | question 2 |" in md
    # The report names no judge model: the warning says the run's judge differs.
    assert "> The run was judged by —" in md

    steady = summary.model_copy(update={"disagreements": [], "run_judge_model": JUDGE})
    md = render_rejudge_markdown(steady)
    assert "_Every verdict agreed._" in md
    assert "> The run was judged by" not in md


# ------------------------------------------------------------------- CLI


def _cli(monkeypatch, capsys, tmp_path, items, *flags: str) -> tuple[int, str]:
    report, _ = _run()
    report_path = tmp_path / "report.json"
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    dataset = tmp_path / "items.jsonl"
    dataset.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
    asked: list[set[int]] = []

    async def load_contexts(ids):
        asked.append(ids)
        return CONTEXTS

    monkeypatch.setattr(runner, "load_contexts", load_contexts)
    monkeypatch.setattr(runner, "llm_service_for", lambda node: (_Provider(), JUDGE, 0.0))
    argv = ["runner", "rejudge", str(report_path), "--dataset", str(dataset), *flags]
    monkeypatch.setattr(sys, "argv", argv)
    code = 0
    try:
        runner.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    assert asked in ([], [{10, 11, 99}])
    return code, capsys.readouterr().err


def test_the_cli_writes_the_agreement_beside_the_report(monkeypatch, capsys, tmp_path):
    _, items = _run()
    code, err = _cli(monkeypatch, capsys, tmp_path, items, "--times", "3")

    assert (code, err) == (0, "")
    (md,) = tmp_path.glob("rejudge_abcdef12_*.md")
    assert "| correctness | 5 | 3 (60.0%) |" in md.read_text(encoding="utf-8")
    saved = eval_schemas.RejudgeSummary.model_validate_json(
        md.with_suffix(".json").read_text(encoding="utf-8")
    )
    assert saved.times == 3


def test_the_cli_refuses_a_dataset_the_run_did_not_read(monkeypatch, capsys, tmp_path):
    _, items = _run()
    code, err = _cli(monkeypatch, capsys, tmp_path, items[:5], "-o", str(tmp_path / "x.md"))
    assert code == runner.EXIT_ABORTED
    assert "item 6 of the run is not in the dataset" in err
    assert not (tmp_path / "x.md").exists()


@pytest.mark.parametrize("times", ["0", "11"])
def test_a_count_out_of_range_is_refused(monkeypatch, capsys, tmp_path, times):
    _, items = _run()
    code, err = _cli(monkeypatch, capsys, tmp_path, items, "--times", times)
    assert code == 2 and "--times" in err


# -------------------------------------------------------------- Postgres

DB_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")
async def test_the_contexts_come_back_by_id_from_the_chunks_table(monkeypatch):
    engine = create_async_engine(DB_URL)  # type: ignore[arg-type]
    async with engine.connect() as conn:
        outer = await conn.begin()
        async with AsyncSession(bind=conn, expire_on_commit=False) as session:
            coll = Collection(name="rejudge-contexts")
            session.add(coll)
            await session.flush()
            doc = Document(
                collection_id=coll.id,
                title="d.md",
                filename="d.md",
                file_type="md",
                raw_text="one two",
            )
            session.add(doc)
            await session.flush()
            chunks = [
                Chunk(
                    document_id=doc.id,
                    chunk_index=i,
                    text=t,
                    token_count=1,
                    embedding=[0.0] * settings.embedding_dimension,
                )
                for i, t in enumerate(["one", "two"])
            ]
            session.add_all(chunks)
            await session.flush()

            class _Session:
                async def __aenter__(self):
                    return session

                async def __aexit__(self, *_exc):
                    return False

            monkeypatch.setattr(runner, "SessionLocal", _Session)
            ids = {c.id for c in chunks}
            gone = max(ids) + 1000

            assert await runner.load_contexts({*ids, gone}) == {
                chunks[0].id: "one",
                chunks[1].id: "two",
            }
            assert await runner.load_contexts(set()) == {}
        await outer.rollback()
    await engine.dispose()
