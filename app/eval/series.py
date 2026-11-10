"""Several runs of one config, read together (audit E7).

One run is one draw: the same commit, dataset and settings land a few
points apart from run to run. A series says how far apart (the mean, the
range and the t interval of the mean), and which items flip between runs,
so a later change is weighed against that spread and not against one draw.
The gate reads the mean.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from app.eval.protocol import mean_interval
from app.eval.report import CHECKS
from app.eval.schemas import EvalReport, FlakyItem, SeriesRun, SeriesSummary, Spread


def summarize(reports: Sequence[EvalReport]) -> SeriesSummary:
    if not reports:
        raise ValueError("A series needs at least one run")
    first = reports[0]
    return SeriesSummary(
        timestamp=datetime.now(UTC),
        model=first.model,
        prompt_version=first.prompt_version,
        judge_prompt_version=first.judge_prompt_version,
        dataset_size=first.dataset_size,
        config=first.config,
        runs=[
            SeriesRun(
                run_id=r.run_id,
                accuracy=r.metrics.accuracy,
                hallucination_rate=r.metrics.hallucination_rate,
                judge_errors=r.metrics.judge_errors,
            )
            for r in reports
        ],
        accuracy=_spread([r.metrics.accuracy for r in reports]),
        hallucination_rate=_spread([r.metrics.hallucination_rate for r in reports]),
        flaky=flaky_items(reports),
    )


def _spread(values: list[float]) -> Spread:
    interval = mean_interval(values)
    return Spread(
        mean=sum(values) / len(values),
        min=min(values),
        max=max(values),
        low=interval[0] if interval else None,
        high=interval[1] if interval else None,
    )


def flaky_items(reports: Sequence[EvalReport]) -> list[FlakyItem]:
    """The items that passed a check in some runs and failed it in others.
    A failed judge call is no verdict either way."""
    questions: dict[int, str] = {}
    verdicts: dict[tuple[int, str], list[bool]] = {}
    for report in reports:
        for r in report.results:
            questions.setdefault(r.item_id, r.question)
            for check, verdict in CHECKS.items():
                v = verdict(r)
                if not v.error:
                    verdicts.setdefault((r.item_id, check), []).append(v.passed)
    return [
        FlakyItem(
            item_id=item_id,
            question=questions[item_id],
            check=check,
            passed=sum(seen),
            judged=len(seen),
        )
        for (item_id, check), seen in sorted(verdicts.items())
        if 0 < sum(seen) < len(seen)
    ]
