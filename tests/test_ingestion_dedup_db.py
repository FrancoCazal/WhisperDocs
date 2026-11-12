"""A collection holds each document once, and every document all its chunks
(audit R12), against a real Postgres.

A chunk whose text another document held too used to be skipped: deleting
that document took the text out of the collection. Now each document stores
every chunk, and a text already embedded by the same model gets its vector
copied. The same document twice in one collection is a 409, decided before
anything is embedded, and the unique index keeps two uploads at once apart.

Needs TEST_DATABASE_URL, a database migrated to head. Each test runs inside a
transaction that is rolled back; the session commits to savepoints in it.
"""

from __future__ import annotations

import os

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.db import get_db
from app.main import app
from app.models import Chunk, Collection, Document
from app.services import ingestion
from app.services.embedder import get_embedder

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")

DIMENSION = 1536
# About 300 tokens each: two of them make a document of two chunks, cut
# between them, since no overlap crosses a paragraph.
INTRO_A = "Alpha team owns the billing service and its alerts. " * 30
INTRO_B = "Beta team owns the search service and its dashboards. " * 30
SHARED = "Every service writes structured logs with a request id. " * 30


class _Embedder:
    model_id = "fake-embedding"

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[1.0] + [0.0] * (DIMENSION - 1) for _ in texts]


@pytest.fixture
async def api():
    engine = create_async_engine(DB_URL)
    embedder = _Embedder()
    async with engine.connect() as conn:
        outer = await conn.begin()
        async with AsyncSession(
            bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
        ) as session:

            async def _db():
                yield session

            app.dependency_overrides[get_db] = _db
            app.dependency_overrides[get_embedder] = lambda: embedder
            transport = httpx.ASGITransport(app=app)
            try:
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test/api/v1"
                ) as client:
                    yield client, session, embedder
            finally:
                app.dependency_overrides.clear()
        await outer.rollback()
    await engine.dispose()


async def _collection(session: AsyncSession, name: str) -> int:
    coll = Collection(name=name)
    session.add(coll)
    await session.commit()
    return coll.id


async def _upload(client, collection_id: int, name: str, text: str) -> httpx.Response:
    return await client.post(
        "/documents/upload",
        files={"file": (name, text.encode(), "text/plain")},
        data={"collection_id": str(collection_id)},
    )


async def _chunk_texts(session: AsyncSession, document_id: int) -> list[str]:
    rows = await session.scalars(
        select(Chunk.text).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
    )
    return list(rows.all())


async def test_the_same_document_twice_in_a_collection_is_a_409(api):
    client, session, embedder = api
    cid = await _collection(session, "dedup-twice")
    other = await _collection(session, "dedup-elsewhere")

    first = await _upload(client, cid, "policy.txt", INTRO_A + "\n\n" + SHARED)
    again = await _upload(client, cid, "policy-copy.txt", INTRO_A + "\n\n" + SHARED)
    texts_embedded = len(embedder.texts)
    elsewhere = await _upload(client, other, "policy.txt", INTRO_A + "\n\n" + SHARED)

    assert first.status_code == 201
    assert again.status_code == 409
    assert f'as "policy.txt" (id {first.json()["id"]})' in again.json()["detail"]
    # Refused before anything was embedded.
    assert texts_embedded == 2
    # Another collection takes it, with the vectors its texts already have.
    assert elsewhere.status_code == 201
    assert len(embedder.texts) == 2
    stored = await session.scalar(
        select(func.count(Document.id)).where(Document.collection_id == cid)
    )
    assert stored == 1


async def test_a_shared_chunk_is_stored_in_both_documents_and_survives_a_delete(api):
    """The data loss of audit R12: B's copy of a text A held was skipped, so
    deleting A took it out of the collection."""
    client, session, embedder = api
    cid = await _collection(session, "dedup-shared")

    a = (await _upload(client, cid, "a.txt", INTRO_A + "\n\n" + SHARED)).json()
    b = (await _upload(client, cid, "b.txt", INTRO_B + "\n\n" + SHARED)).json()

    assert (a["chunk_count"], b["chunk_count"]) == (2, 2)
    # SHARED was embedded once, for A; B's copy reused its vector.
    assert sum(text.startswith("Every service") for text in embedder.texts) == 1
    assert len(embedder.texts) == 3
    models = await session.scalars(
        select(Chunk.embedding_model).where(Chunk.document_id.in_([a["id"], b["id"]]))
    )
    assert set(models.all()) == {"fake-embedding"}

    assert (await client.delete(f"/documents/{a['id']}")).status_code == 204

    texts = await _chunk_texts(session, b["id"])
    assert len(texts) == 2 and texts[1] == SHARED.strip()


async def test_two_uploads_at_once_are_kept_apart_by_the_index(api, monkeypatch):
    """The check before embedding cannot see an upload that has not committed
    yet: the unique index refuses the second, and it is a 409, not a 500."""
    client, session, _ = api
    cid = await _collection(session, "dedup-race")
    # As the extractor gives it back: without the trailing space.
    text = INTRO_A.strip()
    held = Document(
        collection_id=cid,
        title="first.txt",
        filename="first.txt",
        file_type="txt",
        raw_text=text,
        content_hash=ingestion.content_hash(text),
    )
    session.add(held)
    await session.commit()
    held_id = held.id

    real = ingestion.ensure_new_document
    checks = 0

    async def misses_it_once(*args) -> None:
        # The check before embedding misses it; the one after the index
        # refused the commit finds the holder.
        nonlocal checks
        checks += 1
        if checks > 1:
            await real(*args)

    # As if the other upload committed between the check and the insert.
    monkeypatch.setattr(ingestion, "ensure_new_document", misses_it_once)

    r = await _upload(client, cid, "second.txt", text)

    assert r.status_code == 409
    assert f'as "first.txt" (id {held_id})' in r.json()["detail"]
    count = await session.scalar(
        select(func.count(Document.id)).where(Document.collection_id == cid)
    )
    assert count == 1


async def test_the_content_hash_is_the_one_postgres_computes(api):
    """Alembic 0021 hashed the stored documents in SQL; new ones are hashed
    in Python. Both must agree, or a document stored before would not be
    found again."""
    _, session, _ = api
    text = "Año y señal: «comillas» y emoji 🚀\nsegunda línea"

    in_sql = await session.scalar(
        select(func.encode(func.sha256(func.convert_to(text, "UTF8")), "hex"))
    )

    assert in_sql == ingestion.content_hash(text)
