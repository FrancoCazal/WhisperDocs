import argparse
import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.eval.agreement import build_labels, measure_agreement, read_labels, write_labels
from app.eval.evaluator import EvalAbortedError, RAGEvaluator
from app.eval.judge_cache import judge_cache_from_settings
from app.eval.judges import Judge
from app.eval.rejudge import DatasetMismatchError, items_for, rejudge
from app.eval.report import (
    diff_failures,
    gate_failures,
    render_agreement_markdown,
    render_badge_json,
    render_diff_markdown,
    render_markdown,
    render_rejudge_markdown,
    render_series_markdown,
    threshold_failures,
)
from app.eval.schemas import (
    AgreementSummary,
    EvalReport,
    LabelRow,
    RejudgeSummary,
    SeriesSummary,
    TestItem,
)
from app.eval.series import summarize
from app.models import Chunk, Collection
from app.services.embedder import EmbeddingService
from app.services.llm import llm_service_for


DEFAULT_DATASET = Path("eval/golden/queries.jsonl")
DEFAULT_RESULTS_DIR = Path("eval/results")

# Exit codes: 2 when the judge failed too often to score the run, or
# `rejudge` got a dataset the run did not read (argparse also exits 2, on a
# bad flag), 3 when a run missed a threshold or a diff found the current run
# worse than the previous one past a threshold.
EXIT_ABORTED = 2
EXIT_BELOW_THRESHOLD = 3

# Each run calls the RAG model once and the judge up to twice per item; so
# does each time `rejudge` judges a run again, without the RAG model.
MAX_RUNS = 20
MAX_REJUDGE_TIMES = 10
MAX_LABELS = 500


def _fraction(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} is not between 0 and 1")
    return number


def _up_to(maximum: int) -> Callable[[str], int]:
    def count(value: str) -> int:
        number = int(value)
        if not 1 <= number <= maximum:
            raise argparse.ArgumentTypeError(f"{value} is not between 1 and {maximum}")
        return number

    return count


def load_dataset(path: Path) -> list[TestItem]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        items = json.loads(raw)
    return [TestItem(**item) for item in items]


async def _resolve_collection_id(collection_id: int | None, collection_name: str | None) -> int:
    if collection_id is not None:
        return collection_id
    if collection_name is None:
        raise ValueError("Either --collection-id or --collection-name must be provided.")
    async with SessionLocal() as db:
        result = await db.execute(select(Collection).where(Collection.name == collection_name))
        coll = result.scalar_one_or_none()
    if coll is None:
        raise ValueError(
            f"Collection {collection_name!r} not found. Run scripts/seed_eval.py first."
        )
    return coll.id


async def run_eval(
    dataset_path: Path,
    results_dir: Path,
    collection_id: int | None,
    collection_name: str | None,
    top_k: int,
    concurrency: int,
    write_markdown: bool = False,
    limit: int | None = None,
    judge_cache: bool = True,
) -> EvalReport:
    dataset = load_dataset(dataset_path)
    if not dataset:
        raise ValueError(f"Dataset at {dataset_path} is empty")
    if limit is not None and limit > 0:
        dataset = dataset[:limit]

    resolved_collection_id = await _resolve_collection_id(collection_id, collection_name)

    embedder = EmbeddingService()
    # The eval answers at its own temperature, not the chat's (audit E7).
    rag_provider, rag_model, _chat_temp = llm_service_for("rag")
    judge_provider, judge_model, judge_temp = llm_service_for("eval_judge")
    cache = judge_cache_from_settings() if judge_cache else None
    judge = Judge(judge_provider, judge_model, judge_temp, cache=cache)

    evaluator = RAGEvaluator(
        session_maker=SessionLocal,
        embedder=embedder,
        provider=rag_provider,
        model=rag_model,
        temperature=settings.agent.eval_rag_temperature,
        judge=judge,
        collection_id=resolved_collection_id,
        top_k=top_k,
        concurrency=concurrency,
    )
    report = await evaluator.run(dataset)

    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.timestamp.strftime('%Y-%m-%dT%H-%M-%S')}_{report.run_id[:8]}"
    output_path = results_dir / f"{stem}.json"
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    if write_markdown:
        md_path = results_dir / f"{stem}.md"
        md_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Markdown report: {md_path}")

    badge_path = results_dir / "latest-badge.json"
    badge_path.write_text(render_badge_json(report), encoding="utf-8")

    print_summary(report, output_path)
    return report


async def run_series(
    runs: int, results_dir: Path, write_markdown: bool = False, **run_args: Any
) -> SeriesSummary:
    """`runs` runs of one config, one after another, each saved as a report,
    then read together (app/eval/series.py). The judge's cache carries over
    between them, so the spread is the answers' and not the judge's."""
    reports: list[EvalReport] = []
    for i in range(runs):
        print(f"\n--- Run {i + 1} of {runs} ---")
        reports.append(
            await run_eval(results_dir=results_dir, write_markdown=write_markdown, **run_args)
        )
    summary = summarize(reports)
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"series_{summary.timestamp.strftime('%Y-%m-%dT%H-%M-%S')}"
    (results_dir / f"{stem}.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    md_path = results_dir / f"{stem}.md"
    md_path.write_text(render_series_markdown(summary), encoding="utf-8")
    print(f"\n=== Series of {runs} runs ===")
    for label, spread in (
        ("Accuracy:          ", summary.accuracy),
        ("Unsupported claims:", summary.hallucination_rate),
    ):
        interval = (
            ""
            if spread.low is None or spread.high is None
            else f", 95% interval {spread.low:.1%} - {spread.high:.1%}"
        )
        print(
            f"{label} mean {spread.mean:.1%} (min {spread.min:.1%}, max {spread.max:.1%}{interval})"
        )
    print(f"Flaky items:        {len(summary.flaky)}")
    print(f"Series written to: {md_path}")
    return summary


def print_summary(report: EvalReport, output_path: Path) -> None:
    m = report.metrics
    print("\n=== Evaluation Report ===")
    print(f"Run ID: {report.run_id}")
    print(f"Timestamp: {report.timestamp.isoformat()}")
    print(f"Model: {report.model}")
    print(f"Dataset size: {report.dataset_size}")
    if report.judge_cache_hits is not None:
        print(f"Judge verdicts from the cache: {report.judge_cache_hits}")
    print()
    print(f"Accuracy:             {m.accuracy:.2%}")
    scoped = m.in_scope_unsupported()
    if scoped is not None:
        print(f"Unsupported claims:   {scoped[0]:.2%} of {scoped[1]} in-scope items (faithfulness)")
    print(f"Unsupported, all:     {m.hallucination_rate:.2%} (the gates' hallucination rate)")
    print(f"Retrieval precision:  {m.avg_retrieval_precision:.2%}")
    print(f"Retrieval recall:     {m.avg_retrieval_recall:.2%}")
    print(
        f"Latency avg/p50/p95:  {m.avg_latency_ms:.0f} / {m.latency_p50_ms:.0f} / {m.latency_p95_ms:.0f} ms"
    )
    print(f"Avg tokens (in/out):  {m.avg_tokens_input:.0f} / {m.avg_tokens_output:.0f}")
    print(f"Total cost estimate:  ${m.estimated_cost_usd:.4f} USD")
    print()
    print("By difficulty:")
    for difficulty, bucket in m.by_difficulty.items():
        print(
            f"  {difficulty:<6}: n={bucket['total']:<3} "
            f"accuracy={bucket['accuracy']:.2%} "
            f"unsupported={bucket['hallucination_rate']:.2%}"
        )
    print()
    print("By category:")
    for category, bucket in m.by_category.items():
        print(
            f"  {category:<14}: n={bucket['total']:<3} "
            f"accuracy={bucket['accuracy']:.2%} "
            f"unsupported={bucket['hallucination_rate']:.2%}"
        )
    print()
    print(f"Full report written to: {output_path}")


def _load_report(path: Path) -> EvalReport:
    return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))


def cmd_diff(args: argparse.Namespace) -> list[str]:
    """Write the diff, and return how the current run is worse than the
    previous one past the thresholds."""
    prev = _load_report(args.previous)
    curr = _load_report(args.current)
    md = render_diff_markdown(prev, curr)
    if args.output:
        args.output.write_text(md, encoding="utf-8")
        print(f"Diff written to: {args.output}")
    else:
        print(md)
    return diff_failures(
        prev,
        curr,
        max_accuracy_drop=args.max_accuracy_drop,
        max_hallucination_rise=args.max_hallucination_rise,
    )


async def load_contexts(chunk_ids: set[int]) -> dict[int, str]:
    """The texts of the chunks that still exist, by id."""
    if not chunk_ids:
        return {}
    async with SessionLocal() as db:
        rows = await db.execute(select(Chunk.id, Chunk.text).where(Chunk.id.in_(chunk_ids)))
    return {chunk_id: text for chunk_id, text in rows.all()}


async def run_rejudge(
    report_path: Path,
    dataset_path: Path,
    times: int,
    concurrency: int,
    output: Path | None = None,
) -> RejudgeSummary:
    """Judge a saved run's answers `times` more times, with no cache, and
    write how often the verdicts agree (app/eval/rejudge.py)."""
    report = _load_report(report_path)
    items = items_for(report, load_dataset(dataset_path))
    contexts = await load_contexts({i for r in report.results for i in (r.context_chunk_ids or [])})
    provider, model, temperature = llm_service_for("eval_judge")
    # No cache: the point is to ask again.
    judge = Judge(provider, model, temperature)
    summary = await rejudge(report, items, contexts, judge, times=times, concurrency=concurrency)
    stamp = summary.timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    md_path = output or report_path.parent / f"rejudge_{report.run_id[:8]}_{stamp}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_rejudge_markdown(summary), encoding="utf-8")
    md_path.with_suffix(".json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n=== Judge consistency, {times} more times ===")
    for c in summary.checks:
        print(
            f"{c.check:<14}: {c.unanimous} of {c.items} items unanimous, "
            f"{c.flipped} of {c.verdicts_compared} new verdicts flipped, {c.skipped} skipped"
        )
    print(f"Written to: {md_path}")
    return summary


async def run_label(
    report_path: Path,
    dataset_path: Path,
    n: int,
    seed: int,
    output: Path | None = None,
) -> list[LabelRow]:
    """Write a sample of a run's verdicts for a person to rule on, without
    what the judge ruled (app/eval/agreement.py). No model is called."""
    report = _load_report(report_path)
    items = items_for(report, load_dataset(dataset_path))
    contexts = await load_contexts({i for r in report.results for i in (r.context_chunk_ids or [])})
    rows = build_labels(report, items, contexts, n=n, seed=seed)
    path = output or report_path.parent / f"labels_{report.run_id[:8]}.jsonl"
    write_labels(path, rows)
    by_check = {c: sum(r.check == c for r in rows) for c in ("correctness", "hallucination")}
    print(
        f"\n{len(rows)} verdicts to rule on ({by_check['correctness']} correctness, "
        f"{by_check['hallucination']} grounding), written to: {path}"
    )
    print('Fill "human" in each line with true or false (see "ask"), then run:')
    print(f"  python -m app.eval.runner agreement {report_path} {path}")
    return rows


def run_agreement(
    report_path: Path, labels_path: Path, output: Path | None = None
) -> AgreementSummary:
    """Set a person's rulings against the judge's verdicts of the run. No
    model is called."""
    report = _load_report(report_path)
    summary = measure_agreement(report, read_labels(labels_path))
    md_path = output or labels_path.with_name(f"agreement_{report.run_id[:8]}.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_agreement_markdown(summary), encoding="utf-8")
    md_path.with_suffix(".json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print("\n=== Judge vs human ===")
    for c in summary.checks:
        kappa = "n/a" if c.kappa is None else f"{c.kappa:.2f}"
        print(f"{c.check:<14}: {c.agree} of {c.labeled} agree, kappa {kappa}")
    print(f"{summary.unlabeled} rows without a ruling, {summary.unmatched} not matched")
    print(f"Written to: {md_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation against the test dataset.")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run evaluation suite.")
    for p in (parser, run_parser):
        p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
        p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
        p.add_argument("--collection-id", type=int, default=None)
        p.add_argument("--collection-name", type=str, default="Eval Golden")
        p.add_argument("--top-k", type=int, default=5)
        p.add_argument("--concurrency", type=int, default=4)
        p.add_argument("--markdown", action="store_true", help="Also write a .md report.")
        p.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Run only the first N dataset items (smoke runs).",
        )
        p.add_argument(
            "--runs",
            type=_up_to(MAX_RUNS),
            default=1,
            help=f"Run the config this many times (1 to {MAX_RUNS}) and write a series "
            "summary: the mean, range and interval, and the items that flip. The "
            "thresholds then apply to the mean.",
        )
        p.add_argument(
            "--no-judge-cache",
            action="store_true",
            help="Judge every answer again instead of reusing the verdicts in "
            "AGENT_EVAL_JUDGE_CACHE.",
        )
        p.add_argument(
            "--min-accuracy",
            type=_fraction,
            default=settings.agent.eval_min_accuracy,
            help=f"Exit {EXIT_BELOW_THRESHOLD} below this accuracy; 0 turns it off "
            "(default: AGENT_EVAL_MIN_ACCURACY, %(default)s).",
        )
        p.add_argument(
            "--max-hallucination",
            type=_fraction,
            default=settings.agent.eval_max_hallucination_rate,
            help=f"Exit {EXIT_BELOW_THRESHOLD} above this hallucination rate; 1 turns it off "
            "(default: AGENT_EVAL_MAX_HALLUCINATION_RATE, %(default)s).",
        )

    diff_parser = sub.add_parser("diff", help="Compare two report JSON files.")
    diff_parser.add_argument("previous", type=Path)
    diff_parser.add_argument("current", type=Path)
    diff_parser.add_argument("-o", "--output", type=Path, default=None)
    diff_parser.add_argument(
        "--max-accuracy-drop",
        type=_fraction,
        default=settings.agent.eval_max_accuracy_drop,
        help=f"Exit {EXIT_BELOW_THRESHOLD} when the current run's accuracy is more than this "
        "below the previous run's; 1 turns it off "
        "(default: AGENT_EVAL_MAX_ACCURACY_DROP, %(default)s).",
    )
    diff_parser.add_argument(
        "--max-hallucination-rise",
        type=_fraction,
        default=settings.agent.eval_max_hallucination_rise,
        help=f"Exit {EXIT_BELOW_THRESHOLD} when the current run's hallucination rate is more "
        "than this above the previous run's; 1 turns it off "
        "(default: AGENT_EVAL_MAX_HALLUCINATION_RISE, %(default)s).",
    )

    rejudge_parser = sub.add_parser(
        "rejudge",
        help="Judge a saved run's answers again, with no cache, to measure how often the "
        "judge agrees with itself. Calls the judge's paid API.",
    )
    rejudge_parser.add_argument("report", type=Path)
    rejudge_parser.add_argument(
        "--times",
        type=_up_to(MAX_REJUDGE_TIMES),
        default=3,
        help=f"How many more times to judge each answer (1 to {MAX_REJUDGE_TIMES}).",
    )
    rejudge_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="The dataset the run read: the judge reads its expected answers.",
    )
    rejudge_parser.add_argument("--concurrency", type=int, default=4)
    rejudge_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Where to write the Markdown; the JSON goes next to it. Default: beside the report.",
    )

    label_parser = sub.add_parser(
        "label",
        help="Write a sample of a run's verdicts for a person to rule on, without what the "
        "judge ruled. Calls no model.",
    )
    label_parser.add_argument("report", type=Path)
    label_parser.add_argument(
        "--n",
        type=_up_to(MAX_LABELS),
        default=50,
        help=f"How many verdicts to sample (1 to {MAX_LABELS}), half of each check and, "
        "within a check, half the judge passed and half it failed.",
    )
    label_parser.add_argument("--seed", type=int, default=0)
    label_parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="The dataset the run read: the person sees the expected answer and facts.",
    )
    label_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Where to write the labels (JSON lines); it must not exist. "
        "Default: beside the report.",
    )

    agreement_parser = sub.add_parser(
        "agreement",
        help="Set a person's rulings (from `label`) against the judge's verdicts of the run: "
        "agreement, Cohen's kappa and the verdicts they split on. Calls no model.",
    )
    agreement_parser.add_argument("report", type=Path)
    agreement_parser.add_argument("labels", type=Path)
    agreement_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Where to write the Markdown; the JSON goes next to it. Default: beside the labels.",
    )

    args = parser.parse_args()

    if args.command == "diff":
        _exit_on(cmd_diff(args), "Worse than the previous run:")
        return
    if args.command == "label":
        try:
            asyncio.run(run_label(args.report, args.dataset, args.n, args.seed, args.output))
        except (DatasetMismatchError, FileExistsError) as exc:
            print(f"\n{exc}", file=sys.stderr)
            sys.exit(EXIT_ABORTED)
        return
    if args.command == "agreement":
        try:
            run_agreement(args.report, args.labels, args.output)
        except ValueError as exc:
            print(f"\n{exc}", file=sys.stderr)
            sys.exit(EXIT_ABORTED)
        return
    if args.command == "rejudge":
        try:
            asyncio.run(
                run_rejudge(args.report, args.dataset, args.times, args.concurrency, args.output)
            )
        except DatasetMismatchError as exc:
            print(f"\n{exc}", file=sys.stderr)
            sys.exit(EXIT_ABORTED)
        return

    run_args: dict[str, Any] = {
        "dataset_path": args.dataset,
        "results_dir": args.results_dir,
        "collection_id": args.collection_id,
        "collection_name": args.collection_name,
        "top_k": args.top_k,
        "concurrency": args.concurrency,
        "write_markdown": args.markdown,
        "limit": args.limit,
        "judge_cache": not args.no_judge_cache,
    }
    try:
        if args.runs > 1:
            summary = asyncio.run(run_series(args.runs, **run_args))
        else:
            report = asyncio.run(run_eval(**run_args))
    except EvalAbortedError as exc:
        print(f"\n{exc}\nFirst judge error: {exc.first_error}", file=sys.stderr)
        sys.exit(EXIT_ABORTED)

    if args.runs > 1:
        failures = threshold_failures(
            summary.accuracy.mean,
            summary.hallucination_rate.mean,
            min_accuracy=args.min_accuracy,
            max_hallucination=args.max_hallucination,
            of="mean ",
        )
        _exit_on(failures, f"Below the thresholds, over {args.runs} runs:")
        return
    failures = gate_failures(
        report.metrics,
        min_accuracy=args.min_accuracy,
        max_hallucination=args.max_hallucination,
    )
    _exit_on(failures, "Below the thresholds:")


def _exit_on(failures: list[str], heading: str) -> None:
    if failures:
        print(f"\n{heading}", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        sys.exit(EXIT_BELOW_THRESHOLD)


if __name__ == "__main__":
    main()
