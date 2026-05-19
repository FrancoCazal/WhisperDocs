from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.db import SessionLocal
from app.eval.evaluator import RAGEvaluator
from app.eval.judges import Judge
from app.eval.schemas import EvalReport, TestItem
from app.services.embedder import EmbeddingService, get_embedder
from app.services.llm import llm_service_for

router = APIRouter(prefix="/eval", tags=["eval"])

RESULTS_DIR = Path("eval/results")


class EvalRunRequest(BaseModel):
    dataset: list[TestItem]
    collection_id: int = 1
    top_k: int = Field(default=5, ge=1, le=20)
    concurrency: int = Field(default=4, ge=1, le=16)


@router.post("/run", response_model=EvalReport)
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
    report = await evaluator.run(payload.dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{report.timestamp.strftime('%Y-%m-%dT%H-%M-%S')}_{report.run_id[:8]}.json"
    (RESULTS_DIR / filename).write_text(report.model_dump_json(indent=2), encoding="utf-8")

    return report


@router.get("/results")
async def list_results(limit: int = 20) -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)[:limit]
    summaries = []
    for f in files:
        try:
            report = EvalReport.model_validate_json(f.read_text(encoding="utf-8"))
            summaries.append(
                {
                    "run_id": report.run_id,
                    "timestamp": report.timestamp.isoformat(),
                    "dataset_size": report.dataset_size,
                    "model": report.model,
                    "accuracy": report.metrics.accuracy,
                    "hallucination_rate": report.metrics.hallucination_rate,
                    "avg_retrieval_precision": report.metrics.avg_retrieval_precision,
                    "avg_retrieval_recall": report.metrics.avg_retrieval_recall,
                    "avg_latency_ms": report.metrics.avg_latency_ms,
                    "latency_p50_ms": report.metrics.latency_p50_ms,
                    "latency_p95_ms": report.metrics.latency_p95_ms,
                    "estimated_cost_usd": report.metrics.estimated_cost_usd,
                    "file": f.name,
                }
            )
        except Exception:
            continue
    return summaries
