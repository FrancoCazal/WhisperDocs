from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.logging_config import get_logger
from app.services.embedder import EmbeddingError, EmbeddingService

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(deep: bool = False, db: AsyncSession = Depends(get_db)) -> dict:
    """Liveness + readiness check.

    Default: DB ping only (fast).
    deep=true: also verifies pgvector extension and embedder connectivity.
    """
    checks: dict[str, str] = {}

    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        logger.exception("health_db_failed")
        checks["database"] = f"error: {exc}"

    if not deep:
        status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
        return {"status": status, "checks": checks}

    try:
        result = await db.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        checks["pgvector"] = "ok" if result.first() else "missing"
    except Exception as exc:
        logger.exception("health_pgvector_failed")
        checks["pgvector"] = f"error: {exc}"

    try:
        embedder = EmbeddingService()
        await embedder.embed("ping")
        checks["embedder"] = "ok"
    except EmbeddingError as exc:
        checks["embedder"] = f"error: {exc}"
    except Exception as exc:
        logger.exception("health_embedder_failed")
        checks["embedder"] = f"error: {exc}"

    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks}
