import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.eval.evaluator import EvalAbortedError, RAGEvaluator
from app.eval.judge_cache import judge_cache_from_settings
from app.eval.judges import Judge
from app.eval.report import (
    diff_failures,
    gate_failures,
    render_badge_json,
    render_diff_markdown,
    render_markdown,
)
from app.eval.schemas import EvalReport, TestItem
from app.models import Collection
from app.services.embedder import EmbeddingService
from app.services.llm import llm_service_for


DEFAULT_DATASET = Path("eval/golden/queries.jsonl")
DEFAULT_RESULTS_DIR = Path("eval/results")

# Exit codes: 2 when the judge failed too often to score the run (argparse
# also exits 2, on a bad flag), 3 when a run missed a threshold or a diff
# found the current run worse than the previous one past a threshold.
EXIT_ABORTED = 2
EXIT_BELOW_THRESHOLD = 3


def _fraction(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} is not between 0 and 1")
    return number


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
    print(f"Hallucination rate:   {m.hallucination_rate:.2%}")
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
            f"hallucination={bucket['hallucination_rate']:.2%}"
        )
    print()
    print("By category:")
    for category, bucket in m.by_category.items():
        print(
            f"  {category:<14}: n={bucket['total']:<3} "
            f"accuracy={bucket['accuracy']:.2%} "
            f"hallucination={bucket['hallucination_rate']:.2%}"
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

    args = parser.parse_args()

    if args.command == "diff":
        _exit_on(cmd_diff(args), "Worse than the previous run:")
        return

    try:
        report = asyncio.run(
            run_eval(
                dataset_path=args.dataset,
                results_dir=args.results_dir,
                collection_id=args.collection_id,
                collection_name=args.collection_name,
                top_k=args.top_k,
                concurrency=args.concurrency,
                write_markdown=args.markdown,
                limit=args.limit,
                judge_cache=not args.no_judge_cache,
            )
        )
    except EvalAbortedError as exc:
        print(f"\n{exc}\nFirst judge error: {exc.first_error}", file=sys.stderr)
        sys.exit(EXIT_ABORTED)

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
