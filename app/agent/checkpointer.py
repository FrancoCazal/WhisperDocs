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

Pruning (audit N6): the saver keeps every checkpoint a run writes, about five
per turn, and each one that changes `messages` stores the whole list again.
A thread's rows grow with the square of its turns: 120 synthetic RAG turns
left 609 checkpoints and 70 MB of `messages` copies, for 0.6 MB of messages.
Nothing here reads a thread's history, only its latest state, so a background
sweep keeps each thread's latest checkpoint and drops the rest
(`AsyncPostgresSaver.aprune`, strategy `keep_latest`).
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.errors import DeadlockDetected, SerializationFailure
from psycopg.rows import tuple_row
from psycopg_pool import AsyncConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None
_pruning: asyncio.Task[None] | None = None

# Threads with a checkpoint beyond the latest of each namespace (the graph has
# no subgraphs, so one namespace): those a turn has run on since the last sweep.
THREADS_TO_PRUNE_SQL = """
SELECT thread_id FROM checkpoints
GROUP BY thread_id
HAVING count(*) > count(DISTINCT checkpoint_ns)
"""


def _redact_dsn(dsn: str) -> str:
    """postgresql://user:secret@host:5432/db -> postgresql://***@host:5432/db"""
    scheme, sep, rest = dsn.partition("://")
    if not sep or "@" not in rest:
        return dsn
    return f"{scheme}://***@{rest.rsplit('@', 1)[1]}"


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
            # Checked on checkout: a connection that died with a Postgres
            # restart is replaced, instead of failing every thread read.
            check=AsyncConnectionPool.check_connection,
            open=False,
        )
        await _pool.open()

        # The annotation wants dict_row connections; the saver sets dict_row
        # on every cursor it opens, so tuple-row connections work the same.
        _checkpointer = AsyncPostgresSaver(conn=_pool)  # type: ignore[arg-type]
        await _checkpointer.setup()
    except Exception:
        # Never log the DSN: it carries the database user and password.
        logger.error(
            "checkpointer_init_failed dsn=%s",
            _redact_dsn(settings.psycopg_database_url),
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


@dataclass(frozen=True)
class Pruned:
    threads: int = 0
    # Changed by another transaction while being pruned (a turn, a delete, a
    # second sweep): left for the next sweep.
    busy: int = 0
    failed: int = 0


@contextlib.asynccontextmanager
async def _connection(
    conn: AsyncConnection[Any] | AsyncConnectionPool[Any],
) -> AsyncIterator[AsyncConnection[Any]]:
    if isinstance(conn, AsyncConnectionPool):
        async with conn.connection() as c:
            yield c
    else:
        yield conn


async def threads_to_prune(saver: AsyncPostgresSaver) -> list[str]:
    """The threads holding more than their latest checkpoint."""
    async with _connection(saver.conn) as conn, conn.cursor(row_factory=tuple_row) as cur:
        await cur.execute(THREADS_TO_PRUNE_SQL)
        return [str(row[0]) for row in await cur.fetchall()]


async def prune_threads(saver: AsyncPostgresSaver, thread_ids: Sequence[str]) -> Pruned:
    """Keep each thread's latest checkpoint, its pending writes and the blobs
    it reads; delete the rest. One thread at a time: each is its own
    transaction, and one that fails leaves the others to go ahead."""
    pruned = busy = failed = 0
    for thread_id in thread_ids:
        try:
            await saver.aprune([thread_id], strategy="keep_latest")
        except (SerializationFailure, DeadlockDetected):
            busy += 1
        except Exception:
            failed += 1
            logger.warning("checkpoint_prune_failed thread_id=%s", thread_id, exc_info=True)
        else:
            pruned += 1
    return Pruned(threads=pruned, busy=busy, failed=failed)


async def prune_checkpoints(saver: AsyncPostgresSaver) -> Pruned:
    """One sweep: prune every thread with more than its latest checkpoint."""
    result = await prune_threads(saver, await threads_to_prune(saver))
    if result != Pruned():
        logger.info(
            "checkpoints_pruned threads=%d busy=%d failed=%d",
            result.threads,
            result.busy,
            result.failed,
        )
    return result


async def _prune_every(saver: AsyncPostgresSaver, interval_s: float) -> None:
    # Waits before the first sweep: startup does not wait on it, and a restart
    # loop does not sweep on every start.
    while True:
        await asyncio.sleep(interval_s)
        try:
            await prune_checkpoints(saver)
        except Exception:
            logger.warning("checkpoint_prune_sweep_failed", exc_info=True)


def start_pruning(saver: AsyncPostgresSaver) -> None:
    """Sweep every `AGENT_CHECKPOINT_PRUNE_INTERVAL_S` seconds; 0 turns it off.
    Call from lifespan at startup."""
    global _pruning
    interval_s = settings.agent.checkpoint_prune_interval_s
    if interval_s <= 0 or (_pruning is not None and not _pruning.done()):
        return
    _pruning = asyncio.create_task(_prune_every(saver, interval_s), name="checkpoint-pruning")


async def stop_pruning() -> None:
    """Cancel the sweep. A thread cut mid-prune rolls back. Call from lifespan
    at shutdown, before `close_checkpointer`."""
    global _pruning
    task, _pruning = _pruning, None
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
