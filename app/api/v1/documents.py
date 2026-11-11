import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import upstream_error
from app.api.rate_limit import uploads_limit
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
from app.services import catalog
from app.services.embedder import EmbeddingError, EmbeddingService, get_embedder
from app.services.ingestion import (
    DuplicateDocumentError,
    FreshChunk,
    chunk_hashes,
    chunk_rows,
    chunk_text_in_thread,
    content_hash,
    create_job,
    duplicate_from,
    embed_chunks,
    ensure_new_document,
    extract_text_in_thread,
    fresh_chunks,
    reusable_embeddings,
    spawn_ingestion,
)
from app.services.text_extractor import (
    EmptyDocumentError,
    UnreadableDocumentError,
    check_content,
    detect_file_type,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

MB = 1024 * 1024
MAX_UPLOAD_BYTES = 10 * MB
MAX_BATCH_FILES = 20
MAX_BATCH_BYTES = 50 * MB
# `documents.filename`, `documents.title` and their ingestion_jobs columns are String(300).
MAX_NAME_LENGTH = 300
_MULTIPART_OVERHEAD = 64 * 1024  # boundaries, part headers and the form fields

# Request body limits, enforced by BodySizeLimitMiddleware before the form is
# parsed. Paths are relative to this router's mount point.
BODY_LIMITS = {
    "/documents/upload": MAX_UPLOAD_BYTES + _MULTIPART_OVERHEAD,
    "/documents/upload-async": MAX_UPLOAD_BYTES + _MULTIPART_OVERHEAD,
    "/documents/upload-batch": MAX_BATCH_BYTES + _MULTIPART_OVERHEAD,
}


class FileTooLargeError(ValueError):
    pass


async def _read_upload(file: UploadFile) -> tuple[str, str, bytes]:
    """Validate an uploaded file and read it. Returns (filename, file_type, bytes).

    Raises a ValueError subclass for every rejection; `_upload_error` maps
    them to status codes. Reads at most one byte past the limit, so an
    oversized part is not loaded whole.
    """
    filename = file.filename or "untitled"
    if len(filename) > MAX_NAME_LENGTH:
        raise ValueError(f"Filename is longer than {MAX_NAME_LENGTH} characters")
    file_type = detect_file_type(filename)
    file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise FileTooLargeError(f"File exceeds maximum size of {MAX_UPLOAD_BYTES // MB} MB")
    check_content(file_bytes, file_type)
    return filename, file_type, file_bytes


def _upload_error(exc: Exception) -> Exception:
    """Status code for an upload rejection. The subclasses go before ValueError.

    The rejections are this module's and the extractor's own messages, fit for
    the client. A failed embedding says why without the provider's text."""
    if isinstance(exc, EmbeddingError):
        return upstream_error(exc, "Embedding the document")
    if isinstance(exc, FileTooLargeError):
        code = status.HTTP_413_CONTENT_TOO_LARGE
    elif isinstance(exc, DuplicateDocumentError):
        code = status.HTTP_409_CONFLICT
    elif isinstance(exc, (EmptyDocumentError, UnreadableDocumentError)):
        code = status.HTTP_422_UNPROCESSABLE_CONTENT
    elif isinstance(exc, ValueError):  # includes UnsupportedFileTypeError
        code = status.HTTP_400_BAD_REQUEST
    else:
        raise exc
    return HTTPException(status_code=code, detail=str(exc))


def _batch_item_error(exc: Exception) -> str:
    """A failed file's message in a batch: the rejection, or for anything else
    a message without the exception's text, which goes to the log."""
    try:
        rejection = _upload_error(exc)
    except Exception:
        logger.exception("batch_upload_item_failed")
        return "The upload failed; the API's logs have the details."
    return str(getattr(rejection, "detail", rejection))


async def _ingest_single(
    file: UploadFile,
    collection_id: int,
    title: str | None,
    db: AsyncSession,
    embedder: EmbeddingService,
) -> Document:
    filename, file_type, file_bytes = await _read_upload(file)
    raw_text = await extract_text_in_thread(file_bytes, file_type)
    chunks_data, language = await chunk_text_in_thread(raw_text, file_type)
    if not chunks_data:
        raise ValueError("Document produced no chunks after processing")

    # The same document twice in a collection is refused before anything is
    # paid for; a chunk another document holds too is stored again, with
    # that chunk's vector (audit R12).
    digest = content_hash(raw_text)
    await ensure_new_document(db, collection_id, digest)
    hashes = chunk_hashes(chunks_data)
    fresh: list[FreshChunk] = fresh_chunks(chunks_data, hashes)
    reused = await reusable_embeddings(db, hashes, embedder.model_id)
    if reused:
        logger.info(
            "chunk_embeddings_reused",
            reused=sum(1 for chunk in fresh if chunk.text_hash in reused),
            chunks=len(fresh),
            collection_id=collection_id,
        )
    embeddings = await embed_chunks(fresh, embedder, reused)

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
    return document


@router.post(
    "/upload",
    operation_id="upload_document",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(uploads_limit)],
)
async def upload_document(
    collection_id: int = Form(...),
    title: str | None = Form(None, max_length=MAX_NAME_LENGTH),
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
    except (ValueError, EmbeddingError) as exc:
        raise _upload_error(exc) from exc


@router.post(
    "/upload-batch",
    operation_id="upload_documents_batch",
    response_model=BatchUploadResponse,
    dependencies=[Depends(uploads_limit)],
)
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

    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At most {MAX_BATCH_FILES} files per batch, got {len(files)}",
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
            results.append(
                BatchUploadItem(status="error", filename=filename, error=_batch_item_error(exc))
            )
            failed += 1

    response.status_code = status.HTTP_207_MULTI_STATUS if failed else status.HTTP_201_CREATED

    return BatchUploadResponse(
        total=len(files),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


@router.get("", operation_id="list_documents", response_model=Page[DocumentRead])
async def list_documents(
    collection_id: int | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> Page[DocumentRead]:
    return await catalog.list_documents(db, collection_id=collection_id, limit=limit, cursor=cursor)


@router.get("/{document_id:int}", operation_id="get_document", response_model=DocumentDetail)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> Document:
    document = await db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


@router.get(
    "/{document_id:int}/chunks", operation_id="list_document_chunks", response_model=list[ChunkRead]
)
async def list_document_chunks(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[Chunk]:
    # Whether it exists, without reading its text (audit R10).
    found = await db.scalar(select(Document.id).where(Document.id == document_id))
    if found is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    result = await db.execute(
        select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
    )
    return list(result.scalars().all())


@router.delete(
    "/{document_id:int}", operation_id="delete_document", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    # One statement; the foreign key removes the chunks (audit R11).
    deleted = await db.scalar(
        delete(Document).where(Document.id == document_id).returning(Document.id)
    )
    if deleted is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    await db.commit()


# ── Bulk operations ──────────────────────────────────────────────────────────


@router.post("/bulk-delete", operation_id="bulk_delete_documents", response_model=BulkOpResult)
async def bulk_delete_documents(
    payload: BulkDocumentIds,
    db: AsyncSession = Depends(get_db),
) -> BulkOpResult:
    """Delete N documents in one statement. The foreign key removes their chunks.

    The statement succeeds or fails as a whole: a database error is a 500 and
    deletes nothing (audit R11).
    """
    if not payload.document_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="document_ids is empty")

    deleted = await db.scalars(
        delete(Document).where(Document.id.in_(payload.document_ids)).returning(Document.id)
    )
    found_ids = set(deleted.all())
    await db.commit()
    return _bulk_result(payload.document_ids, found_ids)


@router.post("/bulk-move", operation_id="bulk_move_documents", response_model=BulkOpResult)
async def bulk_move_documents(
    payload: BulkMoveRequest,
    db: AsyncSession = Depends(get_db),
) -> BulkOpResult:
    """Reassign N documents to another collection, in one statement (see
    `bulk_delete_documents`)."""
    if not payload.document_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="document_ids is empty")

    target = await db.get(Collection, payload.target_collection_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target collection {payload.target_collection_id} not found",
        )

    # A collection holds each document once (audit R12): one whose text the
    # target already holds, or that another moved document brings first,
    # stays where it is.
    rows = (
        await db.execute(
            select(Document.id, Document.content_hash, Document.collection_id)
            .where(Document.id.in_(payload.document_ids))
            .order_by(Document.id)
        )
    ).all()
    held = dict(
        (
            await db.execute(
                select(Document.content_hash, Document.title).where(
                    Document.collection_id == payload.target_collection_id,
                    Document.content_hash.is_not(None),
                    Document.id.not_in(payload.document_ids),
                )
            )
        ).all()
    )
    errors: dict[int, str] = {}
    movable: list[int] = []
    # Who keeps each text among the requested documents.
    claimed: dict[str, int] = {}
    # The documents already in the target come first: they stay, and claim their text.
    for document_id, digest, collection_id in sorted(
        rows, key=lambda row: (row.collection_id != payload.target_collection_id, row.id)
    ):
        if digest is not None and digest in held:
            errors[document_id] = (
                f'The target collection already holds this document, as "{held[digest]}".'
            )
            continue
        if digest is not None and digest in claimed:
            errors[document_id] = (
                f"Document {claimed[digest]}, also in this request, has the same text."
            )
            continue
        if digest is not None:
            claimed[digest] = document_id
        if collection_id != payload.target_collection_id:
            movable.append(document_id)

    if movable:
        try:
            await db.execute(
                update(Document)
                .where(Document.id.in_(movable))
                .values(collection_id=payload.target_collection_id)
            )
        except IntegrityError as exc:
            # An upload of the same text reached the target in between.
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A document with the same text reached the target collection while "
                "moving; nothing was moved. Try again.",
            ) from exc
    await db.commit()
    moved = {document_id for document_id, _digest, _coll in rows} - set(errors)
    return _bulk_result(payload.document_ids, moved, errors)


def _bulk_result(
    requested: list[int], found_ids: set[int], errors: dict[int, str] | None = None
) -> BulkOpResult:
    errors = errors or {}
    not_found = [i for i in requested if i not in found_ids and i not in errors]
    return BulkOpResult(
        requested=len(requested),
        succeeded=len(found_ids),
        failed=len(not_found) + len(errors),
        not_found=not_found,
        errors=errors,
    )


# ── Async ingestion ──────────────────────────────────────────────────────────


@router.post(
    "/upload-async",
    operation_id="upload_document_async",
    response_model=IngestionJobCreate,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(uploads_limit)],
)
async def upload_document_async(
    collection_id: int = Form(...),
    title: str | None = Form(None, max_length=MAX_NAME_LENGTH),
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
        filename, file_type, file_bytes = await _read_upload(file)
    except ValueError as exc:
        raise _upload_error(exc) from exc

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


@router.get("/jobs/{job_id}", operation_id="get_ingestion_job", response_model=IngestionJobRead)
async def get_ingestion_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
) -> IngestionJob:
    job = await db.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


@router.get("/jobs", operation_id="list_ingestion_jobs", response_model=list[IngestionJobRead])
async def list_ingestion_jobs(
    collection_id: int | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> list[IngestionJob]:
    query = select(IngestionJob).order_by(IngestionJob.created_at.desc()).limit(limit)
    if collection_id is not None:
        query = query.where(IngestionJob.collection_id == collection_id)
    result = await db.execute(query)
    return list(result.scalars().all())


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


@router.get("/jobs/{job_id}/stream", operation_id="stream_ingestion_job")
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
