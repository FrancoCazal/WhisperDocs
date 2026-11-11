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
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from typing import NamedTuple

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.exc import IntegrityError

from app.db import SessionLocal
from app.logging_config import get_logger
from app.models import Chunk, Document, IngestionJob, IngestionStatus
from app.services.chunker import Chunk as TextChunk
from app.services.chunker import TextChunker
from app.services.embedder import EmbeddingError, EmbeddingService
from app.services.instruction_scan import instruction_flags
from app.services.lang import detect_language
from app.services.llm.errors import failure_text
from app.services.text_extractor import (
    HEADING_TYPES,
    PAGED_TYPES,
    EmptyDocumentError,
    UnreadableDocumentError,
    UnsupportedFileTypeError,
    detect_file_type,
    extract_text,
)

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


def _chunk_and_detect(raw_text: str, file_type: str) -> tuple[list[TextChunk], str | None]:
    chunks = TextChunker().chunk(
        raw_text, headings=file_type in HEADING_TYPES, paged=file_type in PAGED_TYPES
    )
    return chunks, detect_language(raw_text)


async def chunk_text_in_thread(raw_text: str, file_type: str) -> tuple[list[TextChunk], str | None]:
    """Chunk the text by the structure its type has, and detect its language,
    off the event loop."""
    return await asyncio.to_thread(_chunk_and_detect, raw_text, file_type)


class FreshChunk(NamedTuple):
    """A chunk to store, and where it sits in its document."""

    index: int
    text: str
    token_count: int
    text_hash: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None


def chunk_hashes(chunks: Sequence[TextChunk]) -> list[str]:
    return [hashlib.sha256(c.text.encode("utf-8")).hexdigest() for c in chunks]


def fresh_chunks(chunks: Sequence[TextChunk], hashes: Sequence[str]) -> list[FreshChunk]:
    """Every chunk of the document, with its text's hash. None is skipped:
    a chunk whose text another document holds too used to be left out, and
    deleting that document took the text out of the collection (audit R12).
    Its embedding is reused instead (`embed_chunks`)."""
    return [
        FreshChunk(c.index, c.text, c.token_count, h, c.section, c.page_start, c.page_end)
        for c, h in zip(chunks, hashes, strict=True)
    ]


# --- one document once per collection (audit R12) ----------------------------------

# The unique index on (collection_id, content_hash), alembic 0021.
DUPLICATE_CONSTRAINT = "uq_documents_collection_content"


class DuplicateDocumentError(ValueError):
    """The collection already holds a document with this exact text."""

    def __init__(self, document_id: int | None = None, title: str | None = None):
        self.document_id = document_id
        held = f' as "{title}" (id {document_id})' if document_id is not None else ""
        super().__init__(
            f"The collection already holds this document{held}. Delete it to upload this "
            "one again, or upload it to another collection."
        )


def content_hash(raw_text: str) -> str:
    """SHA-256 of a document's extracted text, which is what two uploads of
    the same file share whatever their names."""
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


async def ensure_new_document(db, collection_id: int, digest: str) -> None:
    """Raise DuplicateDocumentError when the collection holds a document with
    this text. Checked before the embeddings are paid for; the unique index
    catches two uploads that pass it at once."""
    rows = await db.execute(
        select(Document.id, Document.title)
        .where(Document.collection_id == collection_id, Document.content_hash == digest)
        .limit(1)
    )
    for document_id, title in rows.all():
        raise DuplicateDocumentError(document_id, title)


def is_duplicate_violation(exc: IntegrityError) -> bool:
    return DUPLICATE_CONSTRAINT in str(exc.orig)


async def duplicate_from(db, exc: IntegrityError, collection_id: int, digest: str) -> Exception:
    """The error for a commit the unique index refused: two uploads of one
    document at once. Anything else is returned as it was."""
    if not is_duplicate_violation(exc):
        return exc
    await db.rollback()
    try:
        await ensure_new_document(db, collection_id, digest)
    except DuplicateDocumentError as duplicate:
        return duplicate
    return DuplicateDocumentError()


# --- embeddings, once per text and model (audit R12, R19) --------------------------


async def reusable_embeddings(db, hashes: Sequence[str], model: str) -> dict[str, list[float]]:
    """The vectors already stored for these texts by `model`, one per text
    hash, from any collection: the same text through the same model gives
    the same vector. Chunks from before `embedding_model` was recorded are
    not reused, since nothing says which model made them."""
    wanted = set(hashes)
    if not wanted:
        return {}
    rows = await db.execute(
        select(Chunk.text_hash, Chunk.embedding)
        .where(Chunk.text_hash.in_(wanted), Chunk.embedding_model == model)
        .ext(distinct_on(Chunk.text_hash))
    )
    return {text_hash: [float(x) for x in vector] for text_hash, vector in rows.all()}


async def embed_chunks(
    chunks: Sequence[FreshChunk],
    embedder: EmbeddingService,
    reused: dict[str, list[float]],
    *,
    batch_size: int | None = None,
    on_batch: Callable[[int], Awaitable[None]] | None = None,
) -> list[list[float]]:
    """Each chunk's vector, in order: the stored one for its text when there
    is one, else computed, once per distinct text. `batch_size` texts go in
    each call (all at once when None), and `on_batch` gets, after each, how
    many chunks have a vector."""
    pending: dict[str, str] = {}
    for chunk in chunks:
        if chunk.text_hash not in reused:
            pending.setdefault(chunk.text_hash, chunk.text)
    items = list(pending.items())
    step = batch_size or max(len(items), 1)
    computed: dict[str, list[float]] = {}
    for start in range(0, len(items), step):
        batch = items[start : start + step]
        vectors = await embedder.embed_batch([text for _hash, text in batch])
        computed.update(zip((text_hash for text_hash, _text in batch), vectors, strict=True))
        if on_batch is not None:
            await on_batch(
                sum(1 for c in chunks if c.text_hash in reused or c.text_hash in computed)
            )
    return [
        computed[c.text_hash] if c.text_hash in computed else reused[c.text_hash] for c in chunks
    ]


def _scan(texts: list[str]) -> list[list[str]]:
    return [instruction_flags(text) for text in texts]


async def chunk_rows(
    document: Document,
    fresh: Sequence[FreshChunk],
    embeddings: Sequence[list[float]],
    embedding_model: str,
) -> list[Chunk]:
    """The document's chunk rows, each with the instruction-like text it
    holds (`instruction_flags`, audit M-5), and the document's count of them.
    Every way in builds them here: the background job, the synchronous upload
    and the eval seed. A flagged document is logged with its chunks' rules.
    Each row records the model that made its vector (audit R19).

    The rules take about a millisecond a chunk, seconds for a large PDF, so
    they run off the event loop, as parsing and chunking do."""
    flags = await asyncio.to_thread(_scan, [chunk.text for chunk in fresh])
    rows = [
        Chunk(
            document_id=document.id,
            chunk_index=chunk.index,
            text=chunk.text,
            token_count=chunk.token_count,
            text_hash=chunk.text_hash,
            embedding=embedding,
            embedding_model=embedding_model,
            language=document.language,
            instruction_flags=chunk_flags,
            section=chunk.section,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )
        for chunk, embedding, chunk_flags in zip(fresh, embeddings, flags, strict=True)
    ]
    flagged = {row.chunk_index: row.instruction_flags for row in rows if row.instruction_flags}
    document.flagged_chunks = len(flagged)
    if flagged:
        logger.warning(
            "instruction_like_text",
            document_id=document.id,
            filename=document.filename,
            chunks=flagged,
        )
    return rows


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
        chunks_data, language = await chunk_text_in_thread(raw_text, file_type)
        if not chunks_data:
            raise ValueError("Document produced no chunks after processing")

        # The same document twice in a collection is refused before anything
        # is paid for; a chunk another document holds too is stored again,
        # with that chunk's vector (audit R12).
        digest = content_hash(raw_text)
        hashes = chunk_hashes(chunks_data)
        fresh = fresh_chunks(chunks_data, hashes)
        embedder = EmbeddingService()
        async with SessionLocal() as db:
            await ensure_new_document(db, collection_id, digest)
            reused = await reusable_embeddings(db, hashes, embedder.model_id)

        def resolved_share(done: int) -> int:
            return 25 + int(60 * done / len(fresh))

        already = sum(1 for chunk in fresh if chunk.text_hash in reused)
        await _set_status(
            job_id,
            status=IngestionStatus.EMBEDDING,
            progress=resolved_share(already),
            total_chunks=len(fresh),
            processed_chunks=already,
        )

        # Batched embeddings with per-batch progress 25 → 85.
        async def progress(done: int) -> None:
            await _set_status(job_id, progress=resolved_share(done), processed_chunks=done)

        embeddings = await embed_chunks(
            fresh, embedder, reused, batch_size=EMBED_BATCH_SIZE, on_batch=progress
        )

        await _set_status(job_id, status=IngestionStatus.PERSISTING, progress=90)

        async with SessionLocal() as db:
            document = Document(
                collection_id=collection_id,
                title=title or filename,
                filename=filename,
                file_type=file_type,
                raw_text=raw_text,
                content_hash=digest,
                chunk_count=len(fresh),
                total_tokens=sum(chunk.token_count for chunk in fresh),
                language=language,
            )
            db.add(document)
            try:
                await db.flush()
                db.add_all(await chunk_rows(document, fresh, embeddings, embedder.model_id))
                await db.commit()
            except IntegrityError as exc:
                raise await duplicate_from(db, exc, collection_id, digest) from exc
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
            error_message=job_error_message(exc),
            completed_at=datetime.now(timezone.utc),
        )


def job_error_message(exc: BaseException) -> str:
    """A failed job's `error_message`, which the API and its stream return:
    the extractor's own rejections and a duplicate document as they are, why
    an embedding failed, and for anything else no exception text, which can
    carry SQL (audit R9)."""
    if isinstance(
        exc,
        UnsupportedFileTypeError
        | EmptyDocumentError
        | UnreadableDocumentError
        | DuplicateDocumentError,
    ):
        return str(exc)
    if isinstance(exc, EmbeddingError):
        return f"Embedding the document failed: {failure_text(exc)}."
    return "Ingestion failed; the API's logs have the details."


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
