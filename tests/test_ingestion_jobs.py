"""Ingestion jobs are not lost: task references, shutdown and restart (audit R5).

The bytes of an upload live only in the process's memory. These tests check
that a running job is held until it ends, that shutdown marks unfinished jobs
failed, and that startup marks failed what a dead process left behind. The
last one needs Postgres: set TEST_DATABASE_URL, as in test_retrieval_db.py.
"""

from __future__ import annotations

import asyncio
import gc
import os
import weakref

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.models import Collection, IngestionJob, IngestionStatus
from app.services import ingestion

DB_URL = os.environ.get("TEST_DATABASE_URL")


def _spawn(job_id: int = 1) -> asyncio.Task:
    return ingestion.spawn_ingestion(
        job_id=job_id, file_bytes=b"x", filename="doc.md", title=None, collection_id=1
    )


@pytest.fixture
def statuses(monkeypatch) -> list[tuple[IngestionStatus, str | None]]:
    recorded: list[tuple[IngestionStatus, str | None]] = []

    async def record(_job_id, **kwargs):
        if kwargs.get("status") is not None:
            recorded.append((kwargs["status"], kwargs.get("error_message")))

    monkeypatch.setattr(ingestion, "_set_status", record)
    return recorded


@pytest.fixture
def extraction_never_ends(monkeypatch):
    async def wait_forever(*_args):
        await asyncio.get_running_loop().create_future()

    monkeypatch.setattr(ingestion, "extract_text_in_thread", wait_forever)


async def test_a_spawned_job_survives_garbage_collection(statuses, extraction_never_ends):
    """The task waits on a future only it references. Without a strong
    reference, the cycle is garbage, and the job vanished mid-way."""
    task_ref = weakref.ref(_spawn())
    await asyncio.sleep(0)  # let it reach the extraction and wait there
    gc.collect()

    task = task_ref()
    assert task is not None and not task.done()
    task.cancel()
    await asyncio.wait([task])
    assert task not in ingestion._running


async def test_shutdown_cancels_unfinished_jobs_and_marks_them_failed(
    statuses, extraction_never_ends
):
    task = _spawn()
    await asyncio.sleep(0)

    assert await ingestion.stop_ingestions(grace_s=0.05) == 1

    assert task.cancelled()
    assert statuses[-1] == (IngestionStatus.FAILED, ingestion.INTERRUPTED_MESSAGE)
    assert not ingestion._running


async def test_shutdown_lets_jobs_that_end_within_the_grace_period_finish(monkeypatch):
    async def quick_job(**_kwargs):
        await asyncio.sleep(0.05)

    monkeypatch.setattr(ingestion, "run_ingestion", quick_job)
    task = _spawn()

    assert await ingestion.stop_ingestions(grace_s=2.0) == 0
    assert task.done() and not task.cancelled()


@pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")
async def test_startup_fails_only_the_unfinished_jobs(monkeypatch):
    engine = create_async_engine(DB_URL)
    async with engine.connect() as conn:
        outer = await conn.begin()

        def session() -> AsyncSession:
            # Bound to the test's transaction: commit() does not end it.
            return AsyncSession(bind=conn, expire_on_commit=False)

        monkeypatch.setattr(ingestion, "SessionLocal", session)
        async with session() as db:
            coll = Collection(name="r5-jobs")
            db.add(coll)
            await db.flush()
            jobs = {
                s: IngestionJob(
                    collection_id=coll.id, filename=f"{s.value}.md", file_type="md", status=s.value
                )
                for s in IngestionStatus
            }
            db.add_all(jobs.values())
            await db.flush()
            ids = {s: job.id for s, job in jobs.items()}

        assert await ingestion.fail_interrupted_jobs() >= 5

        async with session() as db:
            for s, job_id in ids.items():
                job = await db.get(IngestionJob, job_id, populate_existing=True)
                if s in (IngestionStatus.COMPLETE, IngestionStatus.FAILED):
                    assert job.status == s.value
                    assert job.error_message is None
                else:
                    assert job.status == IngestionStatus.FAILED.value
                    assert job.error_message == ingestion.INTERRUPTED_MESSAGE
                    assert job.completed_at is not None
        await outer.rollback()
    await engine.dispose()
