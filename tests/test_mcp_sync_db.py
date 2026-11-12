"""The MCP connector (audit M-6) against a real Postgres: a first run adds
the server's resources, a second leaves them alone, a changed one is
replaced with its unchanged chunks' vectors, and a dry run writes nothing.
The server is an in-memory MCPServer; only the embedder is replaced.

Needs a database migrated to head (see test_retrieval_db.py); skipped when
TEST_DATABASE_URL is not set. Each test runs inside a transaction that is
rolled back.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import pytest
from mcp import Client
from mcp.server import MCPServer
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.connectors import mcp_sync
from app.models import Chunk, Collection, Document
from app.services.ingestion import content_hash

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")

GUIDE = "file:///docs/guide.md"
PRICES = "file:///docs/prices.csv"
PHOTO = "file:///docs/photo.png"


def _guide(second: str = "beta") -> str:
    # Two sections long enough to be a chunk each (audit R14).
    return f"# Guide\n\n## One\n\n{'alpha ' * 300}\n\n## Two\n\n{(second + ' ') * 300}\n"


@pytest.fixture
async def session():
    engine = create_async_engine(DB_URL)
    async with engine.connect() as conn:
        outer = await conn.begin()
        async with AsyncSession(
            bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
        ) as s:
            yield s
        await outer.rollback()
    await engine.dispose()


class _Embedder:
    model_id = "fake-embedding"
    dimension = 1536

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, text):
        return [0.0] * 1535 + [1.0]

    async def embed_batch(self, texts):
        self.texts.extend(texts)
        return [[0.0] * 1535 + [1.0] for _ in texts]


def _reader(files: dict, uri: str):
    def read() -> str | bytes:
        return files[uri][0]

    return read


class Source:
    """A server whose resources are `files`, read at each call, so a test
    can change one between runs."""

    def __init__(self, files: dict[str, tuple[str | bytes, str]]) -> None:
        self.files = files
        self.server = MCPServer(name="docs")
        for uri, (_content, mime) in files.items():
            self.server.resource(uri, name=uri.rsplit("/", 1)[-1], mime_type=mime)(
                _reader(files, uri)
            )

        @self.server.resource("file:///broken/{name}", mime_type="text/plain")
        def broken(name: str) -> str:
            raise ValueError(f"cannot read {name}")


class World:
    def __init__(self, session, collection_id, embedder):
        self.session = session
        self.collection_id = collection_id
        self.embedder = embedder
        self.source = Source(
            {
                GUIDE: (_guide(), "text/markdown"),
                PRICES: ("item,price\npen,2\nbook,15\n", "text/plain"),
                PHOTO: (b"\x89PNG\r\n", "image/png"),
            }
        )

    @asynccontextmanager
    async def sessions(self):
        yield self.session

    async def run(self, **kwargs) -> mcp_sync.SyncReport:
        async with Client(self.source.server) as client:
            return await mcp_sync.sync(
                client,
                collection_id=self.collection_id,
                sessions=self.sessions,
                embedder=self.embedder,
                **kwargs,
            )

    async def synced(self) -> dict[str, tuple[int, str]]:
        rows = await self.session.execute(
            select(Document.source_uri, Document.id, Document.title).where(
                Document.collection_id == self.collection_id
            )
        )
        return {uri: (document_id, title) for uri, document_id, title in rows.all()}


@pytest.fixture
async def world(session):
    collection = Collection(name="mcp-sync", description="")
    session.add(collection)
    await session.flush()
    return World(session, collection.id, _Embedder())


def _outcomes(report: mcp_sync.SyncReport) -> dict[str, str]:
    return {item.uri: item.outcome for item in report.items}


async def test_a_first_run_adds_the_resources_and_a_second_leaves_them_alone(world):
    first = await world.run()
    embedded = list(world.embedder.texts)
    second = await world.run()

    assert _outcomes(first) == {GUIDE: "added", PRICES: "added", PHOTO: "skipped"}
    (photo,) = [i for i in first.items if i.uri == PHOTO]
    assert photo.detail == "image/png: not a format WhisperDocs reads"
    synced = await world.synced()
    assert set(synced) == {GUIDE, PRICES}
    assert synced[GUIDE][1] == "guide.md"
    file_type = await world.session.scalar(
        select(Document.file_type).where(Document.id == synced[PRICES][0])
    )
    assert file_type == "csv"
    assert _outcomes(second) == {GUIDE: "unchanged", PRICES: "unchanged", PHOTO: "skipped"}
    assert second.counts() == {"unchanged": 2, "skipped": 1}
    # Nothing was embedded again.
    assert world.embedder.texts == embedded
    assert await world.synced() == synced


async def test_a_changed_resource_replaces_its_document_and_keeps_unchanged_vectors(world):
    await world.run(uris=[GUIDE])
    (old_id, _title) = (await world.synced())[GUIDE]
    world.embedder.texts.clear()
    world.source.files[GUIDE] = (_guide(second="gamma"), "text/markdown")

    report = await world.run(uris=[GUIDE])

    (item,) = report.items
    assert item.outcome == "updated"
    new_id, _title = (await world.synced())[GUIDE]
    assert new_id == item.document_id != old_id
    gone = await world.session.scalar(select(func.count(Document.id)).where(Document.id == old_id))
    assert gone == 0
    # Only the section that changed was embedded.
    assert len(world.embedder.texts) == 1
    assert "gamma" in world.embedder.texts[0]
    texts = await world.session.scalars(select(Chunk.text).where(Chunk.document_id == new_id))
    assert any("gamma" in t for t in texts.all())


async def test_a_resource_the_collection_holds_already_is_skipped_naming_it(world):
    text = "item,price\npen,2\nbook,15\n"
    # The text the CSV extractor gives back, uploaded before.
    extracted = (await mcp_sync.extract_document("prices.csv", "csv", text.encode())).raw_text
    held = Document(
        collection_id=world.collection_id,
        title="uploaded.csv",
        filename="uploaded.csv",
        file_type="csv",
        raw_text=extracted,
        content_hash=content_hash(extracted),
    )
    world.session.add(held)
    await world.session.flush()

    report = await world.run(uris=[PRICES])

    (item,) = report.items
    assert (item.outcome, item.document_id) == ("skipped", held.id)
    assert f'as "uploaded.csv" (id {held.id})' in item.detail


async def test_a_dry_run_compares_and_writes_nothing(world):
    before = await world.run(dry_run=True)
    await world.run(uris=[GUIDE])
    world.source.files[GUIDE] = (_guide(second="delta"), "text/markdown")
    world.embedder.texts.clear()
    synced = await world.synced()

    after = await world.run(uris=[GUIDE], dry_run=True)

    assert _outcomes(before) == {GUIDE: "added", PRICES: "added", PHOTO: "skipped"}
    assert [i.detail for i in before.items if i.uri == GUIDE] == ["2 chunks"]
    assert _outcomes(after) == {GUIDE: "updated"}
    assert after.items[0].document_id == synced[GUIDE][0]
    assert world.embedder.texts == []
    assert await world.synced() == synced


async def test_a_resource_the_server_cannot_read_fails_alone(world):
    report = await world.run(uris=["file:///broken/x", PRICES])

    assert _outcomes(report) == {"file:///broken/x": "failed", PRICES: "added"}
    assert report.items[0].detail.startswith("the server could not read it")


async def test_a_collection_that_does_not_exist_stops_the_run(world):
    async with Client(world.source.server) as client:
        with pytest.raises(mcp_sync.SyncError, match="No collection 2000000000"):
            await mcp_sync.sync(
                client,
                collection_id=2_000_000_000,
                sessions=world.sessions,
                embedder=world.embedder,
            )
