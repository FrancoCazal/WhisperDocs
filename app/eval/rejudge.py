"""How consistent the judge is (audit E7): judge a saved run's answers
again, several times, with no cache, and count how often the verdicts
agree.

A verdict the judge flips on the same answer is noise in every metric it
feeds, and a difference between two runs smaller than that noise says
nothing. Only verdicts the judge gave count: an answer that did what an
injection asked fails by rule, and one with no context is grounded by rule.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from app.eval.judges import JUDGE_PROMPT_VERSION, Judge, obeyed_injection
from app.eval.protocol import context_hash, dataset_hash
from app.eval.report import CHECKS
from app.eval.schemas import (
    CheckAgreement,
    Disagreement,
    EvalReport,
    EvalResult,
    JudgeResult,
    RejudgeSummary,
    TestItem,
)


class DatasetMismatchError(ValueError):
    """The dataset given is not the one the run read."""


def items_for(report: EvalReport, dataset: Sequence[TestItem]) -> list[TestItem]:
    """The dataset's items the run answered, in its order. The correctness
    judge reads the expected answer and the text an injection asks for,
    which the report does not keep; they come from the dataset the run
    read, and its hash says whether it is that one."""
    by_id = {item.id: item for item in dataset}
    items: list[TestItem] = []
    for r in report.results:
        item = by_id.get(r.item_id)
        if item is None or item.question != r.question:
            raise DatasetMismatchError(f"item {r.item_id} of the run is not in the dataset")
        items.append(item)
    if report.config is not None and dataset_hash(items) != report.config.dataset_hash:
        raise DatasetMismatchError(
            f"the dataset is not the one the run read ({dataset_hash(items)}, "
            f"not {report.config.dataset_hash})"
        )
    return items


async def rejudge(
    report: EvalReport,
    items: Sequence[TestItem],
    contexts: Mapping[int, str],
    judge: Judge,
    *,
    times: int,
    concurrency: int = 4,
) -> RejudgeSummary:
    """Judge each answer of `report` `times` more times. `contexts` maps a
    chunk id to its text. An answer whose chunks are not all there, or one
    from a report older than the chunk ids, has its grounding left out."""
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(call: Coroutine[Any, Any, JudgeResult]) -> JudgeResult:
        async with semaphore:
            return await call

    jobs: list[tuple[EvalResult, str, Coroutine[Any, Any, JudgeResult]]] = []
    skipped = dict.fromkeys(CHECKS, 0)
    for r, item in zip(report.results, items, strict=True):
        if obeyed_injection(item, r.system_answer) is None:
            jobs += [
                (r, "correctness", judge.judge_correctness(item, r.system_answer))
                for _ in range(times)
            ]
        else:
            skipped["correctness"] += 1
        texts = judged_context(r, contexts)
        if texts:
            jobs += [
                (r, "hallucination", judge.judge_hallucination(r.system_answer, texts))
                for _ in range(times)
            ]
        else:
            skipped["hallucination"] += 1
    verdicts = await asyncio.gather(*(bounded(call) for _, _, call in jobs))

    again: dict[tuple[int, str], list[JudgeResult]] = {}
    results: dict[int, EvalResult] = {}
    for (r, check, _), verdict in zip(jobs, verdicts, strict=True):
        again.setdefault((r.item_id, check), []).append(verdict)
        results[r.item_id] = r
    return RejudgeSummary(
        run_id=report.run_id,
        timestamp=datetime.now(UTC),
        times=times,
        judge_model=judge.model,
        judge_prompt_version=JUDGE_PROMPT_VERSION,
        run_judge_model=report.config.judge_model if report.config else None,
        run_judge_prompt_version=report.judge_prompt_version,
        checks=[_agreement(check, again, results, skipped[check]) for check in CHECKS],
        disagreements=[
            Disagreement(
                item_id=item_id,
                question=results[item_id].question,
                check=check,
                original=_vote(CHECKS[check](results[item_id])),
                again=[_vote(v) for v in verdicts_again],
            )
            for (item_id, check), verdicts_again in sorted(again.items())
            if len(_votes(CHECKS[check](results[item_id]), verdicts_again)) > 1
        ],
    )


def judged_context(r: EvalResult, contexts: Mapping[int, str]) -> list[str] | None:
    """The chunks the judge read for `r`, as they were; None when there were
    none, a chunk is gone or its text changed under the same id."""
    ids = r.context_chunk_ids
    if not ids or any(i not in contexts for i in ids):
        return None
    texts = [contexts[i] for i in ids]
    return texts if context_hash(texts) == r.context_hash else None


def _vote(verdict: JudgeResult) -> bool | None:
    return None if verdict.error else verdict.passed


def _votes(original: JudgeResult, again: list[JudgeResult]) -> set[bool]:
    return {v.passed for v in (original, *again) if not v.error}


def _agreement(
    check: str,
    again: Mapping[tuple[int, str], list[JudgeResult]],
    results: Mapping[int, EvalResult],
    skipped: int,
) -> CheckAgreement:
    items = unanimous = compared = flipped = errors = 0
    for (item_id, name), verdicts in again.items():
        if name != check:
            continue
        original = CHECKS[check](results[item_id])
        errors += sum(v.error for v in verdicts)
        given = [v for v in verdicts if not v.error]
        # Every call failed: nothing to agree or disagree with.
        if not given:
            continue
        items += 1
        unanimous += len(_votes(original, given)) == 1
        if not original.error:
            compared += len(given)
            flipped += sum(v.passed != original.passed for v in given)
    return CheckAgreement(
        check=check,
        items=items,
        unanimous=unanimous,
        verdicts_compared=compared,
        flipped=flipped,
        judge_errors=errors,
        skipped=skipped,
    )
