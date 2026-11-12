"""The MCP server's tools that write (BE-36) against a real Postgres: a key
that writes, the user's confirmation before a deletion or a move, and a row
in `audit_log` for every call. Over the in-memory client, which reaches what
stdio reaches, and over Streamable HTTP with keys, both protocol eras.

Only the embedder is replaced. Needs a database migrated to head (see
test_retrieval_db.py); skipped when TEST_DATABASE_URL is not set. Each test
runs inside a transaction that is rolled back.
"""

from __future__ import annotations

import base64
import os
import re
from contextlib import asynccontextmanager

import httpx
import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import ElicitResult
from pydantic import SecretStr
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.api.rate_limit import RateLimiter
from app.config import settings
from app.db import get_db
from app.main import app
from app.mcp.http import McpOverHttp
from app.mcp.keys import create_key, list_keys
from app.mcp.server import build_server
from app.mcp.writes import READ_ONLY_KEY
from app.models import AuditLog, Chunk, Collection, Document
from app.services.ingestion import content_hash

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")

URL = "http://localhost:8000/mcp"
MASTER = "the-api-key"
RUNBOOK = "# Runbook\n\nRestart the ingestion worker before you rotate the keys."


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
        self.texts.append(text)
        return [0.0] * 1535 + [1.0]

    async def embed_batch(self, texts):
        self.texts.extend(texts)
        return [[0.0] * 1535 + [1.0] for _ in texts]


class _Asker:
    """An elicitation callback: answers `action`, with `confirm` when it
    accepts, and keeps each question."""

    def __init__(self, action: str = "accept", confirm: bool = True) -> None:
        self.action = action
        self.confirm = confirm
        self.questions: list[str] = []

    async def __call__(self, _context, params):
        self.questions.append(params.message)
        content = {"confirm": self.confirm} if self.action == "accept" else None
        return ElicitResult(action=self.action, content=content)


class World:
    def __init__(self, session, a, b, documents, server, endpoint, embedder):
        self.session = session
        self.a = a
        self.b = b
        self.documents = documents
        self.server = server
        self.endpoint = endpoint
        self.embedder = embedder

    def local(self, asker: _Asker | None = None, mode: str = "auto") -> Client:
        """The in-memory client: no HTTP request, so it reaches what stdio does."""
        return Client(self.server, elicitation_callback=asker, mode=mode)

    @asynccontextmanager
    async def http(self, key: str, asker: _Asker | None = None, mode: str = "auto"):
        http = httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=self.endpoint),
            headers={"Authorization": f"Bearer {key}"},
        )
        transport = streamable_http_client(URL, http_client=http)
        async with (
            self.endpoint.running(),
            http,
            Client(transport, mode=mode, elicitation_callback=asker) as c,
        ):
            yield c

    async def documents_in(self, collection_id: int) -> list[str]:
        rows = await self.session.scalars(
            select(Document.title)
            .where(Document.collection_id == collection_id)
            .order_by(Document.id)
        )
        return list(rows.all())

    async def exists(self, document_id: int) -> bool:
        count = await self.session.scalar(
            select(func.count(Document.id)).where(Document.id == document_id)
        )
        return count == 1

    async def log(self) -> list[AuditLog]:
        rows = await self.session.scalars(select(AuditLog).order_by(AuditLog.id))
        return list(rows.all())


@pytest.fixture
async def world(session, monkeypatch):
    """Two collections of a document each, the server on the session, and the
    API with its own key."""
    monkeypatch.setattr(settings, "api_key", SecretStr(MASTER))
    monkeypatch.setattr("app.mcp.server.limiter", RateLimiter())

    collections, documents = {}, {}
    for name in ("a", "b"):
        collection = Collection(name=f"mcp-writes-{name}", description="")
        session.add(collection)
        await session.flush()
        text = f"pgvector in {name}"
        document = Document(
            collection_id=collection.id,
            title=f"{name}.md",
            filename=f"{name}.md",
            file_type="md",
            raw_text=text,
            content_hash=content_hash(text),
            chunk_count=1,
            total_tokens=3,
            language="en",
        )
        session.add(document)
        await session.flush()
        session.add(
            Chunk(
                document_id=document.id,
                chunk_index=0,
                text=text,
                token_count=3,
                embedding=[0.0] * 1535 + [1.0],
                language="en",
            )
        )
        collections[name], documents[name] = collection.id, document.id
    await session.flush()

    @asynccontextmanager
    async def sessions():
        yield session

    embedder = _Embedder()
    server = build_server(sessions=sessions, embedder=lambda: embedder)
    endpoint = McpOverHttp(server=server, sessions=sessions)
    return World(session, collections["a"], collections["b"], documents, server, endpoint, embedder)


def _text(result) -> str:
    """A tool's error, without the SDK's "Error executing tool <name>: "."""
    assert result.is_error, result.structured_content
    return re.sub(r"^Error executing tool \w+: ", "", result.content[0].text)


def _entry(row: AuditLog) -> tuple:
    return (row.actor, row.channel, row.action, row.outcome)


# ---------- ingest_document ----------


async def test_a_document_comes_in_as_an_upload_does_and_once(world):
    async with world.local() as client:
        added = await client.call_tool(
            "ingest_document",
            {"collection_id": world.a, "filename": "runbook.md", "content": RUNBOOK},
        )
        again = await client.call_tool(
            "ingest_document",
            {"collection_id": world.a, "filename": "copy.md", "content": RUNBOOK},
        )

    result = added.structured_content
    assert not added.is_error
    assert result["outcome"] == "done"
    document = result["document"]
    assert (document["title"], document["file_type"], document["chunk_count"]) == (
        "runbook.md",
        "md",
        1,
    )
    assert document["uri"] == f"whisperdocs://documents/{document['id']}"
    stored = await world.session.scalar(
        select(Chunk.text).where(Chunk.document_id == document["id"])
    )
    assert "Restart the ingestion worker" in stored
    # A collection holds each document once.
    assert f'as "runbook.md" (id {document["id"]})' in _text(again)
    done, failed = await world.log()
    assert _entry(done) == ("stdio", "mcp-stdio", "ingest_document", "done")
    assert (done.collection_id, done.document_id) == (world.a, document["id"])
    assert done.detail == {"filename": "runbook.md", "title": "runbook.md", "chunks": 1}
    assert _entry(failed)[3] == "failed"
    assert "already holds this document" in failed.detail["reason"]


async def test_a_binary_file_comes_in_base64_and_bad_base64_is_refused(world):
    payload = base64.b64encode(b"Rotate the keys every quarter.").decode()

    async with world.local() as client:
        added = await client.call_tool(
            "ingest_document",
            {
                "collection_id": world.a,
                "filename": "keys.txt",
                "content": payload,
                "base64": True,
                "title": "Key rotation",
            },
        )
        broken = await client.call_tool(
            "ingest_document",
            {"collection_id": world.a, "filename": "x.txt", "content": "@@@", "base64": True},
        )
        unsupported = await client.call_tool(
            "ingest_document",
            {"collection_id": world.a, "filename": "x.exe", "content": "MZ"},
        )

    assert added.structured_content["document"]["title"] == "Key rotation"
    assert _text(broken) == "`content` is not valid base64."
    assert "exe" in _text(unsupported)
    assert [row.outcome for row in await world.log()] == ["done", "failed", "failed"]


async def test_a_collection_that_does_not_exist_is_refused_before_anything_is_paid(world):
    async with world.local() as client:
        result = await client.call_tool(
            "ingest_document",
            {"collection_id": 2_000_000_000, "filename": "a.md", "content": RUNBOOK},
        )

    assert _text(result).startswith("Collection 2000000000 not found")
    assert world.embedder.texts == []
    (row,) = await world.log()
    assert (row.outcome, row.detail["reason"]) == ("refused", "not found")


# ---------- delete_document ----------


@pytest.mark.parametrize(
    ("asker", "outcome"),
    [
        (_Asker("accept"), "done"),
        (_Asker("accept", confirm=False), "declined"),
        (_Asker("decline"), "declined"),
        (_Asker("cancel"), "cancelled"),
    ],
)
async def test_a_deletion_asks_the_user_and_their_answer_decides(world, asker, outcome):
    document_id = world.documents["a"]

    async with world.local(asker) as client:
        result = await client.call_tool("delete_document", {"document_id": document_id})

    assert not result.is_error
    assert result.structured_content["outcome"] == outcome
    (question,) = asker.questions
    assert question == (
        f'Delete "a.md" (document {document_id}, 1 chunks) from the collection '
        '"mcp-writes-a"? It cannot be undone.'
    )
    assert await world.exists(document_id) is (outcome != "done")
    (row,) = await world.log()
    assert _entry(row) == ("stdio", "mcp-stdio", "delete_document", outcome)
    assert row.document_id == document_id
    if outcome == "done":
        assert row.detail == {"title": "a.md", "chunks": 1}
        assert row.collection_id == world.a


async def test_a_client_that_cannot_ask_its_user_deletes_nothing(world):
    document_id = world.documents["a"]

    async with world.local() as client:
        result = await client.call_tool("delete_document", {"document_id": document_id})

    assert "cannot ask its user" in _text(result)
    assert "Documents page" in _text(result)
    assert await world.exists(document_id)
    (row,) = await world.log()
    assert (row.outcome, row.document_id, row.collection_id) == ("refused", document_id, world.a)


async def test_a_2025_client_with_a_channel_is_asked_in_the_middle_of_the_call(world):
    """MCP 2025-11-25, as over stdio: the question is a request to the
    client while the call waits."""
    asker = _Asker()

    async with world.local(asker, mode="legacy") as client:
        result = await client.call_tool("delete_document", {"document_id": world.documents["a"]})

    assert result.structured_content["outcome"] == "done"
    assert len(asker.questions) == 1
    assert not await world.exists(world.documents["a"])


async def test_deleting_a_document_that_does_not_exist_asks_nothing(world):
    asker = _Asker()

    async with world.local(asker) as client:
        result = await client.call_tool("delete_document", {"document_id": 2_000_000_000})

    assert _text(result).startswith("Document 2000000000 not found")
    assert asker.questions == []


# ---------- move_document ----------


async def test_a_move_asks_the_user_then_moves(world):
    asker = _Asker()
    document_id = world.documents["a"]

    async with world.local(asker) as client:
        result = await client.call_tool(
            "move_document", {"document_id": document_id, "target_collection_id": world.b}
        )

    assert result.structured_content["outcome"] == "done"
    assert result.structured_content["document"]["collection_id"] == world.b
    assert 'to "mcp-writes-b"?' in asker.questions[0]
    assert await world.documents_in(world.b) == ["a.md", "b.md"]
    (row,) = await world.log()
    assert _entry(row)[2:] == ("move_document", "done")
    assert (row.collection_id, row.detail["target_collection_id"]) == (world.a, world.b)


async def test_a_move_the_target_cannot_take_is_refused_before_asking(world):
    asker = _Asker()
    twin = Document(
        collection_id=world.b,
        title="a-copy.md",
        filename="a-copy.md",
        file_type="md",
        raw_text="pgvector in a",
        content_hash=content_hash("pgvector in a"),
    )
    world.session.add(twin)
    await world.session.flush()

    async with world.local(asker) as client:
        held = await client.call_tool(
            "move_document",
            {"document_id": world.documents["a"], "target_collection_id": world.b},
        )
        same = await client.call_tool(
            "move_document",
            {"document_id": world.documents["b"], "target_collection_id": world.b},
        )

    assert (
        _text(held) == 'The collection "mcp-writes-b" already holds this document, as "a-copy.md".'
    )
    assert _text(same).endswith('is in the collection "mcp-writes-b" already.')
    assert asker.questions == []
    assert [row.outcome for row in await world.log()] == ["refused", "refused"]


# ---------- keys over HTTP ----------


async def test_a_key_that_only_reads_writes_nothing(world):
    key = await create_key(world.session, "reader", [world.a])

    async with world.http(key, _Asker()) as client:
        ingest = await client.call_tool(
            "ingest_document",
            {"collection_id": world.a, "filename": "r.md", "content": RUNBOOK},
        )
        delete = await client.call_tool("delete_document", {"document_id": world.documents["a"]})

    assert _text(ingest) == _text(delete) == READ_ONLY_KEY
    assert await world.exists(world.documents["a"])
    assert [_entry(row) for row in await world.log()] == [
        ("reader", "mcp-http", "ingest_document", "refused"),
        ("reader", "mcp-http", "delete_document", "refused"),
    ]


async def test_a_key_that_writes_does_so_only_where_it_reaches(world):
    key = await create_key(world.session, "writer-a", [world.a], can_write=True)

    async with world.http(key, _Asker()) as client:
        into_b = await client.call_tool(
            "ingest_document",
            {"collection_id": world.b, "filename": "r.md", "content": RUNBOOK},
        )
        delete_b = await client.call_tool("delete_document", {"document_id": world.documents["b"]})
        move_to_b = await client.call_tool(
            "move_document",
            {"document_id": world.documents["a"], "target_collection_id": world.b},
        )
        into_a = await client.call_tool(
            "ingest_document",
            {"collection_id": world.a, "filename": "r.md", "content": RUNBOOK},
        )

    # Out of reach reads as missing; the log keeps why.
    assert _text(into_b).startswith(f"Collection {world.b} not found")
    assert _text(delete_b).startswith(f"Document {world.documents['b']} not found")
    assert _text(move_to_b).startswith(f"Collection {world.b} not found")
    assert into_a.structured_content["outcome"] == "done"
    assert await world.documents_in(world.b) == ["b.md"]
    rows = await world.log()
    assert [row.detail.get("reason") for row in rows[:3]] == ["out of the key's reach"] * 3
    assert _entry(rows[3]) == ("writer-a", "mcp-http", "ingest_document", "done")
    (info,) = [k for k in await list_keys(world.session) if k.name == "writer-a"]
    assert info.can_write


async def test_over_stateless_http_the_question_rides_the_tool_result(world):
    """MCP 2026-07-28: the server answers the call with its question; the
    client calls again with the answer and the sealed state."""
    key = await create_key(world.session, "writer", None, can_write=True)
    asker = _Asker()

    async with world.http(key, asker) as client:
        result = await client.call_tool("delete_document", {"document_id": world.documents["b"]})

    assert result.structured_content["outcome"] == "done"
    assert len(asker.questions) == 1
    assert not await world.exists(world.documents["b"])
    (row,) = await world.log()
    assert _entry(row) == ("writer", "mcp-http", "delete_document", "done")


async def test_a_2025_client_over_http_cannot_be_asked(world):
    """MCP 2025-11-25 asks within a session, which the stateless /mcp keeps
    none of."""
    asker = _Asker()

    async with world.http(MASTER, asker, mode="legacy") as client:
        result = await client.call_tool("delete_document", {"document_id": world.documents["b"]})

    assert "cannot carry the question" in _text(result)
    assert asker.questions == []
    assert await world.exists(world.documents["b"])
    (row,) = await world.log()
    assert _entry(row) == ("api-key", "mcp-http", "delete_document", "refused")


# ---------- the log ----------


async def test_the_api_lists_the_log_newest_first(world):
    async with world.local() as client:
        await client.call_tool(
            "ingest_document",
            {"collection_id": world.a, "filename": "runbook.md", "content": RUNBOOK},
        )
        await client.call_tool("delete_document", {"document_id": world.documents["a"]})

    async def db():
        yield world.session

    app.dependency_overrides[get_db] = db
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
            r = await c.get("/admin/audit-log", params={"limit": 1}, headers={"X-API-Key": MASTER})
            rest = await c.get(
                "/admin/audit-log",
                params={"cursor": r.json()["next_cursor"]},
                headers={"X-API-Key": MASTER},
            )
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200
    page = r.json()
    assert page["total"] == 2
    assert [(e["action"], e["outcome"]) for e in page["items"]] == [("delete_document", "refused")]
    assert [(e["action"], e["outcome"]) for e in rest.json()["items"]] == [
        ("ingest_document", "done")
    ]
