"""Async ingestion worker. Processes upload in background, emits progress to IngestionJob row.

Stages: extracting → chunking → embedding (per-batch progress) → persisting → complete.
On failure, status set to "failed" with error_message; partial document/chunks rolled back.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone

from app.db import SessionLocal
from app.logging_config import get_logger
from app.models import Chunk, Document, IngestionJob, IngestionStatus
from app.services.chunker import TextChunker
from app.services.embedder import EmbeddingService
from app.services.lang import detect_language
from app.services.text_extractor import detect_file_type, extract_text

logger = get_logger(__name__)

EMBED_BATCH_SIZE = 32


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
        raw_text = extract_text(file_bytes, file_type)

        await _set_status(job_id, status=IngestionStatus.CHUNKING, progress=15)
        chunker = TextChunker()
        chunks_data = chunker.chunk(raw_text)
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
                language=detect_language(raw_text),
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
    """Fire-and-forget background task. Returns the Task so callers can keep a ref."""
    return asyncio.create_task(
        run_ingestion(
            job_id=job_id,
            file_bytes=file_bytes,
            filename=filename,
            title=title,
            collection_id=collection_id,
        ),
        name=f"ingestion-{job_id}",
    )
