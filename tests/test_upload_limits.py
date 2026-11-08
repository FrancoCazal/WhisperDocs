"""Upload limits and input validation (audit R8 and R18).

Runs the real app over ASGI, without its lifespan, with the database and the
embedder replaced: every request here is rejected before either is needed.
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from app.api.v1 import documents
from app.config import settings
from app.db import get_db
from app.main import app
from app.models import Collection
from app.schemas.query import QueryRequest
from app.services.embedder import get_embedder

UPLOAD_LIMIT = documents.BODY_LIMITS["/documents/upload"]


class _NoRows:
    def scalars(self):
        return self

    def all(self):
        return []


class _FakeDb:
    async def get(self, model, ident):
        return Collection(id=ident, name="c") if model is Collection else None

    async def execute(self, *_args, **_kwargs):
        return _NoRows()


@pytest.fixture
async def client():
    async def fake_db():
        yield _FakeDb()

    app.dependency_overrides[get_db] = fake_db
    app.dependency_overrides[get_embedder] = lambda: object()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
        yield c
    app.dependency_overrides.clear()


def _upload(name: str, content: bytes, mime: str = "application/octet-stream") -> dict:
    return {"files": {"file": (name, content, mime)}, "data": {"collection_id": "1"}}


# --- body size, before the form is parsed --------------------------------------


async def test_declared_body_over_the_limit_is_rejected_up_front(client):
    origin = settings.cors_allowed_origins[0]
    r = await client.post(
        "/documents/upload",
        content=b"x",  # never read: the declared length decides
        headers={
            "content-type": "multipart/form-data; boundary=b",
            "content-length": str(UPLOAD_LIMIT + 1),
            "origin": origin,
        },
    )
    assert r.status_code == 413
    assert "MB limit" in r.json()["detail"]
    # CORS wraps the limit, so a browser sees the 413 and not a CORS error.
    assert r.headers["access-control-allow-origin"] == origin


async def test_chunked_body_is_cut_once_it_passes_the_limit(client):
    boundary = "b"
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="collection_id"\r\n\r\n1\r\n'
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="a.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    body = head + b"a" * UPLOAD_LIMIT + tail
    chunks_sent = 0

    async def stream():
        nonlocal chunks_sent
        for i in range(0, len(body), 64 * 1024):
            chunks_sent += 1
            yield body[i : i + 64 * 1024]

    r = await client.post(
        "/documents/upload",
        content=stream(),
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    assert "content-length" not in r.request.headers
    assert r.status_code == 413


# --- per file ---------------------------------------------------------------------


async def test_file_over_the_limit_is_413(client):
    r = await client.post(
        "/documents/upload-async",
        **_upload("big.txt", b"a" * (documents.MAX_UPLOAD_BYTES + 1), "text/plain"),
    )
    assert r.status_code == 413


async def test_pdf_extension_without_pdf_content_is_400(client):
    r = await client.post("/documents/upload-async", **_upload("fake.pdf", b"MZ\x90\x00 not a pdf"))
    assert r.status_code == 400
    assert "not a PDF" in r.json()["detail"]


async def test_binary_content_with_text_extension_is_400(client):
    r = await client.post("/documents/upload-async", **_upload("notes.txt", b"ab\x00cd" * 10))
    assert r.status_code == 400
    assert "binary" in r.json()["detail"]


async def test_filename_longer_than_the_column_is_400_not_500(client):
    r = await client.post("/documents/upload-async", **_upload("a" * 297 + ".txt", b"hello"))
    assert r.status_code == 400
    assert "300 characters" in r.json()["detail"]


async def test_title_longer_than_the_column_is_422(client):
    payload = _upload("doc.txt", b"hello")
    payload["data"]["title"] = "t" * 301
    r = await client.post("/documents/upload-async", **payload)
    assert r.status_code == 422


async def test_document_without_text_is_422(client):
    """EmptyDocumentError is a ValueError; the old except order turned it into 400."""
    r = await client.post("/documents/upload", **_upload("blank.txt", b" \n\t \n", "text/plain"))
    assert r.status_code == 422


async def test_damaged_pdf_is_422_not_500(client):
    r = await client.post(
        "/documents/upload", **_upload("broken.pdf", b"%PDF-1.4\n" + b"\xff" * 200)
    )
    assert r.status_code == 422
    assert "Could not read the PDF" in r.json()["detail"]


async def test_batch_with_too_many_files_is_400(client):
    files = [("files", (f"f{i}.txt", b"hi", "text/plain")) for i in range(21)]
    r = await client.post("/documents/upload-batch", files=files, data={"collection_id": "1"})
    assert r.status_code == 400
    assert "At most 20 files" in r.json()["detail"]


# --- R18: list limits and session_id ---------------------------------------------


async def test_job_listing_is_not_taken_by_the_document_route(client):
    """`GET /documents/{document_id}` used to match `/documents/jobs` first and
    answer 422 for "jobs" not being an integer, whatever the query."""
    r = await client.get("/documents/jobs")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.parametrize(
    "path", ["/documents", "/documents/jobs", "/query/history", "/eval/results"]
)
@pytest.mark.parametrize("limit", [0, -1, 201])
async def test_list_limit_out_of_range_is_422(client, path, limit):
    r = await client.get(path, params={"limit": limit})
    assert r.status_code == 422


async def test_history_session_id_longer_than_the_column_is_422(client):
    r = await client.get("/query/history", params={"session_id": "s" * 51})
    assert r.status_code == 422


def test_query_session_id_longer_than_the_column_is_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="q", session_id="s" * 51)
    assert QueryRequest(question="q", session_id="s" * 50).session_id == "s" * 50
