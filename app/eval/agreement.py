"""Does the judge agree with a person? (audit E4)

An eval is as good as its judge, and nothing says how often this one rules
as a careful reader would. `build_labels` draws a sample of a run's verdicts
and writes what the judge read, without what it ruled, for a person to rule
on blind. `measure_agreement` then sets the person's rulings against the
judge's: how often they agree, Cohen's kappa, and the verdicts they split on.

The sample is balanced by the judge's verdict. A run passes about nine items
in ten, so a plain sample is nine tenths verdicts that are easy to agree on,
and says little about the failures, which are what a metric is moved by. The
price is that the agreement rate of the sample is not the agreement rate of
the run; the confusion counts, and what a person says of each kind of
verdict, are not distorted by it.

Only verdicts the judge gave count: an answer that did what an injection
asked fails by rule, one with no context is grounded by rule, and a failed
call is no verdict.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from app.eval.judges import obeyed_injection
from app.eval.rejudge import judged_context
from app.eval.report import CHECKS
from app.eval.schemas import (
    AgreementSummary,
    EvalReport,
    HumanAgreement,
    HumanDisagreement,
    LabelRow,
    TestItem,
)

ASKS = {
    "correctness": "Is the system answer correct? Answer true or false.",
    "hallucination": (
        "Is every claim in the system answer supported by the context? Answer true or false."
    ),
}


def _allocate(total: int, sizes: Sequence[int]) -> list[int]:
    """Split `total` as evenly as the sizes allow: each group gets an equal
    share, and what a small group cannot take goes to the others."""
    take = [0] * len(sizes)
    remaining = min(total, sum(sizes))
    while remaining > 0:
        open_groups = [i for i, size in enumerate(sizes) if take[i] < size]
        share, extra = divmod(remaining, len(open_groups))
        given = 0
        for position, i in enumerate(open_groups):
            add = min(share + (1 if position < extra else 0), sizes[i] - take[i])
            take[i] += add
            given += add
        remaining -= given
    return take


def build_labels(
    report: EvalReport,
    items: Sequence[TestItem],
    contexts: Mapping[int, str],
    *,
    n: int,
    seed: int = 0,
) -> list[LabelRow]:
    """Up to `n` verdicts of the run for a person to rule on, half of each
    check, and within a check half the judge passed and half it failed. The
    same report, dataset, `n` and `seed` draw the same sample, in an order
    that does not show which verdicts are which."""
    candidates: dict[str, list[tuple[LabelRow, bool]]] = {check: [] for check in CHECKS}
    for r, item in zip(report.results, items, strict=True):
        if not r.correctness.error and obeyed_injection(item, r.system_answer) is None:
            row = LabelRow(
                run_id=report.run_id,
                item_id=r.item_id,
                check="correctness",
                ask=ASKS["correctness"],
                question=r.question,
                expected_answer=item.expected_answer,
                required_facts=item.required_facts,
                optional_facts=item.optional_facts,
                system_answer=r.system_answer,
            )
            candidates["correctness"].append((row, r.correctness.passed))
        texts = judged_context(r, contexts)
        if texts and not r.hallucination.error:
            row = LabelRow(
                run_id=report.run_id,
                item_id=r.item_id,
                check="hallucination",
                ask=ASKS["hallucination"],
                question=r.question,
                system_answer=r.system_answer,
                context=texts,
            )
            candidates["hallucination"].append((row, r.hallucination.passed))

    rng = random.Random(seed)
    quotas = _allocate(n, [len(found) for found in candidates.values()])
    chosen: list[LabelRow] = []
    for quota, found in zip(quotas, candidates.values(), strict=True):
        passed = [row for row, ok in found if ok]
        failed = [row for row, ok in found if not ok]
        rng.shuffle(passed)
        rng.shuffle(failed)
        from_passed, from_failed = _allocate(quota, [len(passed), len(failed)])
        chosen += passed[:from_passed] + failed[:from_failed]
    rng.shuffle(chosen)
    return chosen


def cohen_kappa(
    both_pass: int, judge_pass_only: int, human_pass_only: int, both_fail: int
) -> float | None:
    """Agreement beyond chance: 1 is perfect, 0 is what two raters with these
    pass rates would reach guessing. None with nothing labeled, or when one
    side never varies and chance already explains it all."""
    n = both_pass + judge_pass_only + human_pass_only + both_fail
    if n == 0:
        return None
    observed = (both_pass + both_fail) / n
    judge_rate = (both_pass + judge_pass_only) / n
    human_rate = (both_pass + human_pass_only) / n
    chance = judge_rate * human_rate + (1 - judge_rate) * (1 - human_rate)
    if chance >= 1:
        return None
    return (observed - chance) / (1 - chance)


def measure_agreement(report: EvalReport, rows: Sequence[LabelRow]) -> AgreementSummary:
    """The labeled rows against the report's verdicts. A row is unmatched when
    it is of another run, names an item or a question the run does not have, or
    the run's verdict was a failed call."""
    results = {r.item_id: r for r in report.results}
    counts = {check: dict.fromkeys(("pp", "pf", "fp", "ff"), 0) for check in CHECKS}
    disagreements: list[HumanDisagreement] = []
    unlabeled = unmatched = 0
    for row in rows:
        if row.human is None:
            unlabeled += 1
            continue
        result = results.get(row.item_id)
        if row.run_id != report.run_id or row.check not in CHECKS or result is None:
            unmatched += 1
            continue
        verdict = CHECKS[row.check](result)
        if verdict.error or row.question != result.question:
            unmatched += 1
            continue
        # Judge first, then person: "pf" is a pass the person would fail.
        key = ("p" if verdict.passed else "f") + ("p" if row.human else "f")
        counts[row.check][key] += 1
        if verdict.passed != row.human:
            disagreements.append(
                HumanDisagreement(
                    item_id=row.item_id,
                    check=row.check,
                    question=result.question,
                    judge_passed=verdict.passed,
                    human_passed=row.human,
                    judge_reasoning=verdict.reasoning,
                    facts_missing=verdict.facts_missing,
                    wrong_claims=verdict.wrong_claims,
                    note=row.note,
                )
            )
    order = list(CHECKS)
    disagreements.sort(key=lambda d: (order.index(d.check), d.item_id))
    checks = []
    for check in CHECKS:
        c = counts[check]
        checks.append(
            HumanAgreement(
                check=check,
                labeled=sum(c.values()),
                agree=c["pp"] + c["ff"],
                both_pass=c["pp"],
                judge_pass_human_fail=c["pf"],
                judge_fail_human_pass=c["fp"],
                both_fail=c["ff"],
                kappa=cohen_kappa(c["pp"], c["pf"], c["fp"], c["ff"]),
            )
        )
    return AgreementSummary(
        run_id=report.run_id,
        timestamp=datetime.now(UTC),
        judge_model=report.config.judge_model if report.config else None,
        judge_prompt_version=report.judge_prompt_version,
        checks=checks,
        disagreements=disagreements,
        unlabeled=unlabeled,
        unmatched=unmatched,
    )


def write_labels(path: Path, rows: Sequence[LabelRow]) -> None:
    """One JSON object per line. Refuses to overwrite a file: it may hold
    hours of someone's rulings."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; move it or pick another file")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")


def read_labels(path: Path) -> list[LabelRow]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(LabelRow.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{path}, line {number}: {exc.errors()[0]['msg']}") from exc
    return rows
