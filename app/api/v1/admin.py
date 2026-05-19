from fastapi import APIRouter

from app.config import settings

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/settings")
async def get_settings() -> dict:
    """Read-only snapshot of agent settings. Useful for UI debugging.

    To change values: set AGENT_* env vars and restart. Runtime mutation
    is intentionally not exposed (would invalidate compiled graph).
    """
    a = settings.agent
    return {
        "models": {
            "classifier": {"model": a.model_classifier, "temperature": a.temp_classifier},
            "query_rewrite": {"model": a.model_query_rewrite, "temperature": a.temp_query_rewrite},
            "rag_generation": {"model": a.model_rag_generation, "temperature": a.temp_rag_generation},
            "clarification": {"model": a.model_clarification, "temperature": a.temp_clarification},
            "answer_direct": {"model": a.model_answer_direct, "temperature": a.temp_answer_direct},
            "summarize": {"model": a.model_summarize, "temperature": a.temp_summarize},
            "eval_judge": {"model": a.model_eval_judge, "temperature": a.temp_eval_judge},
        },
        "retrieval": {
            "top_k": a.retrieval_top_k,
        },
        "compaction": {
            "token_threshold": a.compaction_token_threshold,
            "count_threshold": a.compaction_count_threshold,
            "k_recent": a.k_recent,
        },
        "embedding": {
            "model": settings.embedding_model,
            "dimension": settings.embedding_dimension,
        },
    }
