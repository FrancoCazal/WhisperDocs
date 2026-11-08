import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.errors import APIError
from app.api.rate_limit import eval_limit
from app.db import SessionLocal
from app.eval.evaluator import EvalAbortedError, RAGEvaluator
from app.eval.judges import Judge
from app.eval.schemas import EvalReport, TestItem
from app.services.embedder import EmbeddingService, get_embedder
from app.services.llm import llm_service_for

router = APIRouter(prefix="/eval", tags=["eval"])

RESULTS_DIR = Path("eval/results")


# Each item calls the model twice; the golden set has 50 (audit E23).
MAX_EVAL_ITEMS = 200


class EvalRunRequest(BaseModel):
    dataset: list[TestItem] = Field(max_length=MAX_EVAL_ITEMS)
    collection_id: int = 1
    top_k: int = Field(default=5, ge=1, le=20)
    concurrency: int = Field(default=4, ge=1, le=16)


@router.post(
    "/run", operation_id="run_eval", response_model=EvalReport, dependencies=[Depends(eval_limit)]
)
async def run_evaluation(
    payload: EvalRunRequest,
    embedder: EmbeddingService = Depends(get_embedder),
) -> EvalReport:
    if not payload.dataset:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Dataset is empty")

    rag_provider, rag_model, rag_temp = llm_service_for("rag")
    judge_provider, judge_model, judge_temp = llm_service_for("eval_judge")
    judge = Judge(judge_provider, judge_model, judge_temp)
    evaluator = RAGEvaluator(
        session_maker=SessionLocal,
        embedder=embedder,
        provider=rag_provider,
        model=rag_model,
        temperature=rag_temp,
        judge=judge,
        collection_id=payload.collection_id,
        top_k=payload.top_k,
        concurrency=payload.concurrency,
    )
    try:
        report = await evaluator.run(payload.dataset)
    except EvalAbortedError as exc:
        # The message carries counts only; the provider error stays in the logs.
        raise APIError(status.HTTP_502_BAD_GATEWAY, "eval_aborted", str(exc)) from exc

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{report.timestamp.strftime('%Y-%m-%dT%H-%M-%S')}_{report.run_id[:8]}.json"
    (RESULTS_DIR / filename).write_text(report.model_dump_json(indent=2), encoding="utf-8")

    return report


# The two readers below are plain `def`: FastAPI runs them in its threadpool,
# so reading and parsing reports (~100 KB each) never blocks the event loop.
@router.get("/results", operation_id="list_eval_results")
def list_results(limit: int = Query(default=20, ge=1, le=200)) -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    summaries: list[dict] = []
    # Newest first. Other JSON files live here too (the badge, agent evals), so
    # the limit counts reports read, not files.
    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        if len(summaries) == limit:
            break
        try:
            report = EvalReport.model_validate_json(f.read_text(encoding="utf-8"))
            summaries.append(
                {
                    "run_id": report.run_id,
                    "timestamp": report.timestamp.isoformat(),
                    "dataset_size": report.dataset_size,
                    "model": report.model,
                    "prompt_version": report.prompt_version,
                    "judge_prompt_version": report.judge_prompt_version,
                    "accuracy": report.metrics.accuracy,
                    "hallucination_rate": report.metrics.hallucination_rate,
                    "avg_retrieval_precision": report.metrics.avg_retrieval_precision,
                    "avg_retrieval_recall": report.metrics.measured("avg_retrieval_recall"),
                    "avg_latency_ms": report.metrics.avg_latency_ms,
                    "latency_p50_ms": report.metrics.measured("latency_p50_ms"),
                    "latency_p95_ms": report.metrics.measured("latency_p95_ms"),
                    "estimated_cost_usd": report.metrics.estimated_cost_usd,
                    "file": f.name,
                }
            )
        except Exception:
            continue
    return summaries


# Only what the report saved: an older one has no recall, no latency
# percentiles and no judge error flag, and their defaults would read as
# measurements (frontend review FR-19).
@router.get(
    "/results/{run_id}",
    operation_id="get_eval_result",
    response_model=EvalReport,
    response_model_exclude_unset=True,
)
def get_result(run_id: str) -> EvalReport:
    """One run in full: every item's answers and judge verdicts."""
    try:
        canonical = str(uuid.UUID(run_id))
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    # Reports are saved as "<timestamp>_<first 8 chars of the run id>.json".
    for f in sorted(RESULTS_DIR.glob(f"*_{canonical[:8]}.json"), reverse=True):
        try:
            report = EvalReport.model_validate_json(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if report.run_id == canonical:
            return report
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
