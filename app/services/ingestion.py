"""Async ingestion worker. Processes upload in background, emits progress to IngestionJob row.

Stages: extracting → chunking → embedding (per-batch progress) → persisting → complete.
On failure, status set to "failed" with error_message; partial document/chunks rolled back.

The file bytes live only in this process's memory, so a job cannot outlive it.
On shutdown, `stop_ingestions` gives running jobs a short grace period, then
cancels them and marks them failed. On startup, `fail_interrupted_jobs` marks
failed whatever a process that died without shutting down left unfinished.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone

from sqlalchemy import func, update

from app.db import SessionLocal
from app.logging_config import get_logger
from app.models import Chunk, Document, IngestionJob, IngestionStatus
from app.services.chunker import Chunk as TextChunk
from app.services.chunker import TextChunker
from app.services.embedder import EmbeddingService
from app.services.lang import detect_language
from app.services.text_extractor import detect_file_type, extract_text

logger = get_logger(__name__)

EMBED_BATCH_SIZE = 32

INTERRUPTED_MESSAGE = (
    "Interrupted: the server stopped before the job finished. Upload the file again."
)
_UNFINISHED = [
    s.value for s in IngestionStatus if s not in (IngestionStatus.COMPLETE, IngestionStatus.FAILED)
]

# The event loop keeps only weak references to tasks, so a task nobody holds
# can be garbage collected before it finishes (audit R5). Running ingestions
# stay here until they end.
_running: set[asyncio.Task] = set()


# Parsing, chunking and language detection are CPU-bound (pdfplumber, tiktoken,
# langdetect). Run inline in an `async def`, a large PDF froze every request,
# chat SSE included, until it finished (audit R4). In a worker thread
# pdfplumber still holds the GIL while it computes, so a parse is not faster,
# but the interpreter switches threads every few milliseconds and the event
# loop keeps serving.


async def extract_text_in_thread(file_bytes: bytes, file_type: str) -> str:
    """`extract_text` off the event loop."""
    return await asyncio.to_thread(extract_text, file_bytes, file_type)


def _chunk_and_detect(raw_text: str) -> tuple[list[TextChunk], str | None]:
    return TextChunker().chunk(raw_text), detect_language(raw_text)


async def chunk_text_in_thread(raw_text: str) -> tuple[list[TextChunk], str | None]:
    """Chunk the text and detect its language, off the event loop."""
    return await asyncio.to_thread(_chunk_and_detect, raw_text)


async def _set_status(
    job_id: int,
    *,
    status: IngestionStatus | None = None,
    progress: int | None = None,
    total_chunks: int | None = None,
    processed_chunks: int | None = None,
    document_id: int | None = None,
    error_message: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> None:
    """Single-statement update so progress polls always see consistent state."""
    async with SessionLocal() as db:
        job = await db.get(IngestionJob, job_id)
        if job is None:
            return
        if status is not None:
            job.status = status.value
        if progress is not None:
            job.progress = progress
        if total_chunks is not None:
            job.total_chunks = total_chunks
        if processed_chunks is not None:
            job.processed_chunks = processed_chunks
        if document_id is not None:
            job.document_id = document_id
        if error_message is not None:
            job.error_message = error_message
        if started_at is not None:
            job.started_at = started_at
        if completed_at is not None:
            job.completed_at = completed_at
        await db.commit()


async def create_job(
    *,
    filename: str,
    file_type: str,
    collection_id: int,
    title: str | None = None,
) -> int:
    async with SessionLocal() as db:
        job = IngestionJob(
            collection_id=collection_id,
            filename=filename,
            file_type=file_type,
            title=title,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        return job.id


async def run_ingestion(
    *,
    job_id: int,
    file_bytes: bytes,
    filename: str,
    title: str | None,
    collection_id: int,
) -> None:
    """Main worker. Each stage updates the IngestionJob row so SSE/poll consumers see progress."""
    await _set_status(
        job_id,
        status=IngestionStatus.EXTRACTING,
        progress=5,
        started_at=datetime.now(timezone.utc),
    )

    try:
        file_type = detect_file_type(filename)
        raw_text = await extract_text_in_thread(file_bytes, file_type)

        await _set_status(job_id, status=IngestionStatus.CHUNKING, progress=15)
        chunks_data, language = await chunk_text_in_thread(raw_text)
        if not chunks_data:
            raise ValueError("Document produced no chunks after processing")

        # Dedup against existing chunks in the same collection.
        hashes = [hashlib.sha256(c.text.encode("utf-8")).hexdigest() for c in chunks_data]
        async with SessionLocal() as db:
            from sqlalchemy import select

            existing_q = (
                select(Chunk.text_hash)
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.collection_id == collection_id)
                .where(Chunk.text_hash.in_(hashes))
            )
            skip_hashes = {row[0] for row in (await db.execute(existing_q)).all()}

        fresh = [
            (cd.index, cd.text, cd.token_count, h)
            for cd, h in zip(chunks_data, hashes, strict=True)
            if h not in skip_hashes
        ]
        if not fresh:
            raise ValueError("All chunks already exist in this collection (full duplicate)")

        await _set_status(
            job_id,
            status=IngestionStatus.EMBEDDING,
            progress=25,
            total_chunks=len(fresh),
            processed_chunks=0,
        )

        # Batched embeddings with per-batch progress 25 → 85.
        embedder = EmbeddingService()
        embeddings: list[list[float]] = []
        n = len(fresh)
        for start in range(0, n, EMBED_BATCH_SIZE):
            batch_texts = [t for _i, t, _tc, _h in fresh[start : start + EMBED_BATCH_SIZE]]
            batch_vectors = await embedder.embed_batch(batch_texts)
            embeddings.extend(batch_vectors)
            done = min(start + EMBED_BATCH_SIZE, n)
            pct = 25 + int(60 * done / n)
            await _set_status(job_id, progress=pct, processed_chunks=done)

        await _set_status(job_id, status=IngestionStatus.PERSISTING, progress=90)

        async with SessionLocal() as db:
            document = Document(
                collection_id=collection_id,
                title=title or filename,
                filename=filename,
                file_type=file_type,
                raw_text=raw_text,
                chunk_count=len(fresh),
                total_tokens=sum(tc for _i, _t, tc, _h in fresh),
                language=language,
            )
            db.add(document)
            await db.flush()
            db.add_all(
                Chunk(
                    document_id=document.id,
                    chunk_index=idx,
                    text=t,
                    token_count=tc,
                    text_hash=h,
                    embedding=embedding,
                    language=document.language,
                )
                for (idx, t, tc, h), embedding in zip(fresh, embeddings, strict=True)
            )
            await db.commit()
            await db.refresh(document)
            doc_id = document.id

        await _set_status(
            job_id,
            status=IngestionStatus.COMPLETE,
            progress=100,
            document_id=doc_id,
            completed_at=datetime.now(timezone.utc),
        )
        logger.info("ingestion_job_complete", job_id=job_id, document_id=doc_id)

    except asyncio.CancelledError:
        logger.warning("ingestion_job_interrupted", job_id=job_id)
        await _set_status(
            job_id,
            status=IngestionStatus.FAILED,
            error_message=INTERRUPTED_MESSAGE,
            completed_at=datetime.now(timezone.utc),
        )
        raise
    except Exception as exc:
        logger.exception("ingestion_job_failed", job_id=job_id)
        await _set_status(
            job_id,
            status=IngestionStatus.FAILED,
            error_message=str(exc),
            completed_at=datetime.now(timezone.utc),
        )


def spawn_ingestion(
    *,
    job_id: int,
    file_bytes: bytes,
    filename: str,
    title: str | None,
    collection_id: int,
) -> asyncio.Task:
    """Start the ingestion in the background. The task is kept until it ends."""
    task = asyncio.create_task(
        run_ingestion(
            job_id=job_id,
            file_bytes=file_bytes,
            filename=filename,
            title=title,
            collection_id=collection_id,
        ),
        name=f"ingestion-{job_id}",
    )
    _running.add(task)
    task.add_done_callback(_running.discard)
    return task


async def stop_ingestions(grace_s: float = 5.0) -> int:
    """On shutdown: wait up to `grace_s` for running jobs, then cancel the rest.

    A cancelled job marks itself failed before it ends. The grace period stays
    under Docker's 10 s stop timeout, so the process is not killed while jobs
    are still being marked. A parse already running in a worker thread cannot
    be cancelled and finishes on its own. Returns how many jobs were cancelled.
    """
    if not _running:
        return 0
    _done, pending = await asyncio.wait(set(_running), timeout=grace_s)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(pending, timeout=2.0)
    return len(pending)


async def fail_interrupted_jobs() -> int:
    """On startup: mark failed the jobs a previous process left unfinished.

    Their bytes were in that process's memory, so they can never finish.
    Assumes one process runs ingestion, as compose and the Dockerfile do. With
    several workers, one starting up would also fail the jobs another one is
    running. Returns how many jobs were marked.
    """
    async with SessionLocal() as db:
        result = await db.execute(
            update(IngestionJob)
            .where(IngestionJob.status.in_(_UNFINISHED))
            .values(
                status=IngestionStatus.FAILED.value,
                error_message=INTERRUPTED_MESSAGE,
                completed_at=func.now(),
            )
        )
        await db.commit()
    return result.rowcount
