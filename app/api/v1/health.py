import time

from fastapi import APIRouter, Depends, Request, Response, Security
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

# The embedder check pays an embedding call: its result is reused for this
# long, a failure for less, so a recovered embedder shows up soon (audit E12).
EMBEDDER_OK_TTL_SECONDS = 300.0
EMBEDDER_ERROR_TTL_SECONDS = 30.0
_embedder_check: tuple[float, str] | None = None


async def _check_embedder() -> str:
    global _embedder_check
    now = time.monotonic()
    if _embedder_check is not None and now < _embedder_check[0]:
        return _embedder_check[1]
    try:
        await EmbeddingService().embed("ping")
        result, ttl = "ok", EMBEDDER_OK_TTL_SECONDS
    except Exception as exc:
        logger.exception("health_embedder_failed")
        result, ttl = f"error: {classify_error(exc)}", EMBEDDER_ERROR_TTL_SECONDS
    _embedder_check = (now + ttl, result)
    return result


@router.get(
    "/health",
    operation_id="check_health",
    responses={503: {"description": "Degraded: the same body, with the failing checks."}},
)
async def health(
    request: Request,
    response: Response,
    deep: bool = False,
    db: AsyncSession = Depends(get_db),
    key: str | None = Security(api_key_header),
) -> dict:
    """Liveness + readiness check.

    Default: DB ping only (fast), without the API key: a probe needs none.
    deep=true: also verifies pgvector extension and embedder connectivity. The
    embedder check pays an embedding call, so it needs the key when API_KEY is
    set, counts against the questions rate limit, and its result is reused for
    5 minutes (30 seconds after a failure). It adds `models`: each role's last
    call, from the record real calls leave ("ok", "failing: <kind>", "unknown"
    with no call yet), without calling them. A failing role makes the status
    "degraded".

    A degraded status answers 503, with the same body, so a load balancer or a
    monitor reads it without parsing (audit E12). A failed check reads
    "error: <kind>" (unavailable, billing, timeout, ..., from classify_error);
    the exception is in the logs, not in this open response (audit R9).
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
        healthy = all(v == "ok" for v in checks.values())
        if not healthy:
            response.status_code = 503
        return {"status": "ok" if healthy else "degraded", "checks": checks}

    try:
        result = await db.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
        checks["pgvector"] = "ok" if result.first() else "missing"
    except Exception as exc:
        logger.exception("health_pgvector_failed")
        checks["pgvector"] = f"error: {classify_error(exc)}"

    checks["embedder"] = await _check_embedder()

    models = {
        role: f"failing: {s['last_failure_kind']}" if s["state"] == "failing" else s["state"]
        for role, s in model_status.snapshot().items()
    }
    healthy = all(v == "ok" for v in checks.values()) and not any(
        v.startswith("failing") for v in models.values()
    )
    if not healthy:
        response.status_code = 503
    return {"status": "ok" if healthy else "degraded", "checks": checks, "models": models}
