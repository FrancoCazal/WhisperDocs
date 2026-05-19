import asyncio
import hashlib
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionLocal, get_db
from app.logging_config import get_logger
from app.models import Chunk, Collection, Document, IngestionJob, IngestionStatus
from app.schemas.chunk import ChunkRead
from app.schemas.document import (
    BatchUploadItem,
    BatchUploadResponse,
    BulkDocumentIds,
    BulkMoveRequest,
    BulkOpResult,
    DocumentDetail,
    DocumentRead,
)
from app.schemas.ingestion_job import IngestionJobCreate, IngestionJobRead
from app.schemas.pagination import Page
from app.services.chunker import TextChunker
from app.services.embedder import EmbeddingError, EmbeddingService, get_embedder
from app.services.ingestion import create_job, spawn_ingestion
from app.services.lang import detect_language
from app.services.text_extractor import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
    detect_file_type,
    extract_text,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


async def _ingest_single(
    file: UploadFile,
    collection_id: int,
    title: str | None,
    db: AsyncSession,
    embedder: EmbeddingService,
) -> Document:
    file_type = detect_file_type(file.filename or "")
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File exceeds maximum size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    raw_text = extract_text(file_bytes, file_type)

    chunker = TextChunker()
    chunks_data = chunker.chunk(raw_text)
    if not chunks_data:
        raise ValueError("Document produced no chunks after processing")

    # Dedup: skip chunks whose hash already exists in this collection.
    hashes = [hashlib.sha256(c.text.encode("utf-8")).hexdigest() for c in chunks_data]
    existing_q = (
        select(Chunk.text_hash)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.collection_id == collection_id)
        .where(Chunk.text_hash.in_(hashes))
    )
    existing_hashes = {row[0] for row in (await db.execute(existing_q)).all()}

    fresh: list[tuple[int, str, int, str]] = [
        (cd.index, cd.text, cd.token_count, h)
        for cd, h in zip(chunks_data, hashes, strict=True)
        if h not in existing_hashes
    ]
    skipped = len(chunks_data) - len(fresh)
    if skipped:
        logger.info(
            "chunk_dedup_skipped",
            skipped=skipped,
            kept=len(fresh),
            collection_id=collection_id,
        )

    if not fresh:
        raise ValueError("All chunks already exist in this collection (full duplicate)")

    embeddings = await embedder.embed_batch([t for _idx, t, _tc, _h in fresh])

    document = Document(
        collection_id=collection_id,
        title=title or (file.filename or "untitled"),
        filename=file.filename or "untitled",
        file_type=file_type,
        raw_text=raw_text,
        chunk_count=len(fresh),
        total_tokens=sum(tc for _idx, _t, tc, _h in fresh),
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
    return document


@router.post("/upload", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    collection_id: int = Form(...),
    title: str | None = Form(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder),
) -> Document:
    collection = await db.get(Collection, collection_id)
    if collection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_id} not found",
        )

    try:
        return await _ingest_single(file, collection_id, title, db, embedder)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post("/upload-batch", response_model=BatchUploadResponse)
async def upload_documents_batch(
    response: Response,
    collection_id: int = Form(...),
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder),
) -> BatchUploadResponse:
    collection = await db.get(Collection, collection_id)
    if collection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_id} not found",
        )

    results: list[BatchUploadItem] = []
    succeeded = 0
    failed = 0

    for file in files:
        filename = file.filename or "untitled"
        try:
            document = await _ingest_single(file, collection_id, None, db, embedder)
            results.append(
                BatchUploadItem(
                    status="success",
                    filename=filename,
                    document=DocumentRead.model_validate(document),
                )
            )
            succeeded += 1
        except Exception as exc:
            await db.rollback()
            results.append(BatchUploadItem(status="error", filename=filename, error=str(exc)))
            failed += 1

    response.status_code = status.HTTP_207_MULTI_STATUS if failed else status.HTTP_201_CREATED

    return BatchUploadResponse(
        total=len(files),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


@router.get("", response_model=Page[DocumentRead])
async def list_documents(
    collection_id: int | None = None,
    limit: int = 50,
    cursor: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> Page[DocumentRead]:
    query = select(Document).order_by(Document.id.desc()).limit(limit + 1)
    if collection_id is not None:
        query = query.where(Document.collection_id == collection_id)
    if cursor is not None:
        query = query.where(Document.id < cursor)
    rows = list((await db.execute(query)).scalars().all())
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].id if has_more and items else None
    return Page[DocumentRead](
        items=[DocumentRead.model_validate(d) for d in items],
        next_cursor=next_cursor,
    )


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> Document:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


@router.get("/{document_id}/chunks", response_model=list[ChunkRead])
async def list_document_chunks(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[Chunk]:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    result = await db.execute(
        select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
    )
    return list(result.scalars().all())


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    await db.delete(document)
    await db.commit()


# ── Bulk operations ──────────────────────────────────────────────────────────


@router.post("/bulk-delete", response_model=BulkOpResult)
async def bulk_delete_documents(
    payload: BulkDocumentIds,
    db: AsyncSession = Depends(get_db),
) -> BulkOpResult:
    """Delete N documents in one round-trip. Cascades to chunks via FK."""
    if not payload.document_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="document_ids is empty")

    rows = (
        await db.execute(select(Document).where(Document.id.in_(payload.document_ids)))
    ).scalars().all()
    found_ids = {d.id for d in rows}
    not_found = [i for i in payload.document_ids if i not in found_ids]

    succeeded = 0
    errors: dict[int, str] = {}
    for doc in rows:
        try:
            await db.delete(doc)
            succeeded += 1
        except Exception as exc:
            errors[doc.id] = str(exc)
    await db.commit()

    return BulkOpResult(
        requested=len(payload.document_ids),
        succeeded=succeeded,
        failed=len(not_found) + len(errors),
        not_found=not_found,
        errors=errors,
    )


@router.post("/bulk-move", response_model=BulkOpResult)
async def bulk_move_documents(
    payload: BulkMoveRequest,
    db: AsyncSession = Depends(get_db),
) -> BulkOpResult:
    """Reassign N documents to another collection."""
    if not payload.document_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="document_ids is empty")

    target = await db.get(Collection, payload.target_collection_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target collection {payload.target_collection_id} not found",
        )

    rows = (
        await db.execute(select(Document).where(Document.id.in_(payload.document_ids)))
    ).scalars().all()
    found_ids = {d.id for d in rows}
    not_found = [i for i in payload.document_ids if i not in found_ids]

    succeeded = 0
    errors: dict[int, str] = {}
    for doc in rows:
        try:
            doc.collection_id = payload.target_collection_id
            succeeded += 1
        except Exception as exc:
            errors[doc.id] = str(exc)
    await db.commit()

    return BulkOpResult(
        requested=len(payload.document_ids),
        succeeded=succeeded,
        failed=len(not_found) + len(errors),
        not_found=not_found,
        errors=errors,
    )


# ── Async ingestion ──────────────────────────────────────────────────────────


@router.post("/upload-async", response_model=IngestionJobCreate, status_code=status.HTTP_202_ACCEPTED)
async def upload_document_async(
    collection_id: int = Form(...),
    title: str | None = Form(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> IngestionJobCreate:
    """Kick off ingestion in the background. Returns a job_id immediately.

    Poll `/documents/jobs/{job_id}` or stream `/documents/jobs/{job_id}/stream`.
    """
    collection = await db.get(Collection, collection_id)
    if collection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection {collection_id} not found",
        )

    try:
        file_type = detect_file_type(file.filename or "")
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    filename = file.filename or "untitled"
    job_id = await create_job(
        filename=filename,
        file_type=file_type,
        collection_id=collection_id,
        title=title,
    )
    spawn_ingestion(
        job_id=job_id,
        file_bytes=file_bytes,
        filename=filename,
        title=title,
        collection_id=collection_id,
    )
    return IngestionJobCreate(job_id=job_id, status=IngestionStatus.PENDING.value)


@router.get("/jobs/{job_id}", response_model=IngestionJobRead)
async def get_ingestion_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
) -> IngestionJob:
    job = await db.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


@router.get("/jobs", response_model=list[IngestionJobRead])
async def list_ingestion_jobs(
    collection_id: int | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> list[IngestionJob]:
    query = select(IngestionJob).order_by(IngestionJob.created_at.desc()).limit(limit)
    if collection_id is not None:
        query = query.where(IngestionJob.collection_id == collection_id)
    result = await db.execute(query)
    return list(result.scalars().all())


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


@router.get("/jobs/{job_id}/stream")
async def stream_ingestion_job(job_id: int) -> StreamingResponse:
    """SSE: emits `progress` on every state change and a terminal `complete` or `error`."""

    async def event_stream() -> AsyncIterator[bytes]:
        last_signature: tuple | None = None
        terminal = {IngestionStatus.COMPLETE.value, IngestionStatus.FAILED.value}
        # Cap loop so a stuck job doesn't leak the connection forever (~10 min).
        for _ in range(1200):
            async with SessionLocal() as db:
                job = await db.get(IngestionJob, job_id)
            if job is None:
                yield _sse("error", {"message": f"Job {job_id} not found"})
                return

            sig = (job.status, job.progress, job.processed_chunks)
            if sig != last_signature:
                yield _sse(
                    "progress",
                    {
                        "job_id": job.id,
                        "status": job.status,
                        "progress": job.progress,
                        "processed_chunks": job.processed_chunks,
                        "total_chunks": job.total_chunks,
                    },
                )
                last_signature = sig

            if job.status in terminal:
                if job.status == IngestionStatus.COMPLETE.value:
                    yield _sse(
                        "complete",
                        {"job_id": job.id, "document_id": job.document_id},
                    )
                else:
                    yield _sse(
                        "error",
                        {"job_id": job.id, "message": job.error_message or "ingestion failed"},
                    )
                return

            await asyncio.sleep(0.5)

        yield _sse("error", {"message": "Stream timeout — job still running"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
