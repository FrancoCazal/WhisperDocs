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
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import APIError, upstream_error
from app.api.rate_limit import questions_limit, uploads_limit
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
from app.schemas.extraction import ExtractionRequest, ExtractionResponse
from app.schemas.ingestion_job import IngestionJobCreate, IngestionJobRead
from app.schemas.pagination import Page
from app.services import catalog, extraction
from app.services.documents import MoveConflictError, delete_documents, move_documents
from app.services.embedder import EmbeddingError, EmbeddingService, get_embedder
from app.services.ingestion import (
    MAX_DOCUMENT_BYTES,
    MAX_NAME_LENGTH,
    DocumentTooLargeError,
    DuplicateDocumentError,
    check_document,
    create_job,
    extract_document,
    spawn_ingestion,
    store_document,
)
from app.services.text_extractor import (
    EmptyDocumentError,
    UnreadableDocumentError,
    detect_file_type,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

MB = 1024 * 1024
MAX_UPLOAD_BYTES = MAX_DOCUMENT_BYTES
MAX_BATCH_FILES = 20
MAX_BATCH_BYTES = 50 * MB
_MULTIPART_OVERHEAD = 64 * 1024  # boundaries, part headers and the form fields

# Request body limits, enforced by BodySizeLimitMiddleware before the form is
# parsed. Paths are relative to this router's mount point.
BODY_LIMITS = {
    "/documents/upload": MAX_UPLOAD_BYTES + _MULTIPART_OVERHEAD,
    "/documents/upload-async": MAX_UPLOAD_BYTES + _MULTIPART_OVERHEAD,
    "/documents/upload-batch": MAX_BATCH_BYTES + _MULTIPART_OVERHEAD,
}

FileTooLargeError = DocumentTooLargeError


async def _read_upload(file: UploadFile) -> tuple[str, str, bytes]:
    """Validate an uploaded file and read it. Returns (filename, file_type, bytes).

    Raises a ValueError subclass for every rejection; `_upload_error` maps
    them to status codes. An unsupported type is refused before the body is
    read, and at most one byte past the limit is read, so an oversized part
    is not loaded whole. The checks are `check_document`'s, which every way
    in makes.
    """
    filename = file.filename or "untitled"
    file_type = detect_file_type(filename)
    file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    return filename, check_document(filename, file_bytes, file_type), file_bytes


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
    extracted = await extract_document(filename, file_type, file_bytes)
    return await store_document(db, embedder, extracted, collection_id=collection_id, title=title)


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


@router.post(
    "/{document_id:int}/extract",
    operation_id="extract_fields",
    response_model=ExtractionResponse,
    dependencies=[Depends(questions_limit)],
)
async def extract_document_fields(
    document_id: int,
    payload: ExtractionRequest,
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder),
) -> ExtractionResponse:
    """The fields a JSON Schema names, out of the document, each with the
    passage and the words that state it (audit M-8). One model call, paid
    for; a field the document does not state is null. Shared with the MCP
    server's `extract_fields` (`app/services/extraction.py`)."""
    try:
        return await extraction.extract_fields(
            db,
            embedder,
            document_id=document_id,
            json_schema=payload.json_schema,
            instructions=payload.instructions,
        )
    except extraction.DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except extraction.SchemaError as exc:
        raise APIError(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_schema", f"The schema: {exc}"
        ) from exc
    except EmbeddingError as exc:
        logger.exception("extraction_search_failed", document_id=document_id)
        raise upstream_error(exc, "Searching the document") from exc
    except extraction.ExtractionError as exc:
        logger.exception("extraction_failed", document_id=document_id)
        raise upstream_error(exc, "Extraction") from exc


@router.delete(
    "/{document_id:int}", operation_id="delete_document", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    # One statement; the foreign key removes the chunks (audit R11).
    if not await delete_documents(db, [document_id]):
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

    found_ids = await delete_documents(db, payload.document_ids)
    await db.commit()
    return _bulk_result(payload.document_ids, found_ids)


@router.post("/bulk-move", operation_id="bulk_move_documents", response_model=BulkOpResult)
async def bulk_move_documents(
    payload: BulkMoveRequest,
    db: AsyncSession = Depends(get_db),
) -> BulkOpResult:
    """Reassign N documents to another collection, in one statement (see
    `bulk_delete_documents`). A document whose text the target already holds
    stays where it is (`move_documents`)."""
    if not payload.document_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="document_ids is empty")

    target = await db.get(Collection, payload.target_collection_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target collection {payload.target_collection_id} not found",
        )

    try:
        result = await move_documents(db, payload.document_ids, payload.target_collection_id)
    except MoveConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    await db.commit()
    return _bulk_result(payload.document_ids, result.moved, result.errors)


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
