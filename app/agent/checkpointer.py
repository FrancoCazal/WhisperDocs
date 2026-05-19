"""AsyncPostgresSaver bootstrap (§3.4).

Uses an explicit AsyncConnectionPool (psycopg v3) rather than
AsyncPostgresSaver.from_conn_string — the latter returns a context manager
whose __aexit__ kills the pool, leaving stored references pointing at a dead
connection. With an explicit pool we control lifecycle via lifespan.

autocommit=True is mandatory: AsyncPostgresSaver.setup() runs
`CREATE TABLE IF NOT EXISTS`, which Postgres rejects inside a transaction
block.

Module globals: singleton resource initialized once in lifespan. DI container
or app.state are alternatives but add complexity without benefit for a
resource that never changes at runtime.
"""

import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None


async def init_checkpointer() -> AsyncPostgresSaver:
    """Initialize pool + checkpointer. Idempotent.

    Call from FastAPI lifespan at startup.
    """
    global _pool, _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    try:
        _pool = AsyncConnectionPool(
            conninfo=settings.psycopg_database_url,
            min_size=settings.agent.checkpointer_pool_min_size,
            max_size=settings.agent.checkpointer_pool_max_size,
            kwargs={"autocommit": True},
            open=False,
        )
        await _pool.open()

        _checkpointer = AsyncPostgresSaver(conn=_pool)
        await _checkpointer.setup()
    except Exception:
        logger.error(
            "checkpointer_init_failed dsn=%s",
            settings.psycopg_database_url,
            exc_info=True,
        )
        if _pool is not None:
            await _pool.close()
        _pool = None
        _checkpointer = None
        raise

    return _checkpointer


async def close_checkpointer() -> None:
    """Close the pool. Call from lifespan at shutdown."""
    global _pool, _checkpointer
    if _pool is not None:
        await _pool.close()
    _pool = None
    _checkpointer = None


def get_checkpointer() -> AsyncPostgresSaver:
    """Return the initialized checkpointer or raise."""
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized — call init_checkpointer() from lifespan")
    return _checkpointer
