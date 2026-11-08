from fastapi import APIRouter, Depends, Request, Security
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import api_key_header, check_api_key
from app.api.rate_limit import check_rate
from app.config import settings
from app.db import get_db
from app.logging_config import get_logger
from app.services.embedder import EmbeddingService
from app.services.llm.errors import classify_error
from app.services.model_status import model_status

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", operation_id="check_health")
async def health(
    request: Request,
    deep: bool = False,
    db: AsyncSession = Depends(get_db),
    key: str | None = Security(api_key_header),
) -> dict:
    """Liveness + readiness check.

    Default: DB ping only (fast), without the API key: a probe needs none.
    deep=true: also verifies pgvector extension and embedder connectivity. It
    pays an embedding call, so it needs the key when API_KEY is set, and counts
    against the questions rate limit (audit E12). It adds `models`: each role's
    last call, from the record real calls leave ("ok", "failing: <kind>",
    "unknown" with no call yet), without calling them. A failing role makes the
    status "degraded".

    A failed check reads "error: <kind>" (unavailable, billing, timeout, ...,
    from classify_error); the exception is in the logs, not in this open
    response (audit R9).
    """
    if deep:
        check_api_key(key)
        check_rate(request, key, "questions", settings.rate_limit_per_minute)

    checks: dict[str, str] = {}

    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        logger.exception("health_db_failed")
        checks["database"] = f"error: {classify_error(exc)}"

    if not deep:
        status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
        return {"status": status, "checks": checks}

    try:
        result = await db.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
        checks["pgvector"] = "ok" if result.first() else "missing"
    except Exception as exc:
        logger.exception("health_pgvector_failed")
        checks["pgvector"] = f"error: {classify_error(exc)}"

    try:
        embedder = EmbeddingService()
        await embedder.embed("ping")
        checks["embedder"] = "ok"
    except Exception as exc:
        logger.exception("health_embedder_failed")
        checks["embedder"] = f"error: {classify_error(exc)}"

    models = {
        role: f"failing: {s['last_failure_kind']}" if s["state"] == "failing" else s["state"]
        for role, s in model_status.snapshot().items()
    }
    healthy = all(v == "ok" for v in checks.values()) and not any(
        v.startswith("failing") for v in models.values()
    )
    return {"status": "ok" if healthy else "degraded", "checks": checks, "models": models}
