"""The MCP server over Streamable HTTP (audit M-1, M-2) against a real
Postgres: keys scoped to collections, over the SDK's own HTTP client and
transport. Only the embedder is replaced, and the model is never called.

Needs a database migrated to head (see test_retrieval_db.py); skipped when
TEST_DATABASE_URL is not set. Each test runs inside a transaction that is
rolled back. The database may hold other collections: the tests look for
their own.
"""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import httpx2
import pytest
from mcp import Client, MCPError
from mcp.client.streamable_http import streamable_http_client
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.api.rate_limit import RateLimiter
from app.config import settings
from app.mcp.access import EVERYTHING, Access, access_for, key_hash
from app.mcp.http import McpOverHttp
from app.mcp.keys import KeyCommandError, create_key, list_keys, revoke_key
from app.mcp.server import build_server
from app.models import ApiKey, Chunk, Collection, Document
from app.services.reranker import NoopReranker

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")

URL = "http://localhost:8000/mcp"
MASTER = "the-api-key"
_EMBEDDING = [0.0] * 1535 + [1.0]


@pytest.fixture
async def session():
    engine = create_async_engine(DB_URL)
    async with engine.connect() as conn:
        outer = await conn.begin()
        # The key commands commit; a savepoint keeps that inside the test.
        async with AsyncSession(
            bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
        ) as s:
            yield s
        await outer.rollback()
    await engine.dispose()


class _Embedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, _text):
        self.calls += 1
        return _EMBEDDING


class World:
    def __init__(self, session, a, b, documents, endpoint, embedder):
        self.session = session
        self.a = a
        self.b = b
        self.documents = documents
        self.endpoint = endpoint
        self.embedder = embedder

    @asynccontextmanager
    async def client(self, key: str | None, mode: str = "auto"):
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        http = httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=self.endpoint), headers=headers
        )
        transport = streamable_http_client(URL, http_client=http)
        # The SDK's task group exits in the task it entered: here, not in the
        # fixture, whose teardown runs in another.
        async with self.endpoint.running(), http, Client(transport, mode=mode) as c:
            yield c

    async def post(self, headers: dict[str, str]) -> httpx2.Response:
        transport = httpx2.ASGITransport(app=self.endpoint)
        async with self.endpoint.running(), httpx2.AsyncClient(transport=transport) as http:
            return await http.post(URL, json={}, headers=headers)


@pytest.fixture
async def world(session, monkeypatch):
    """Two collections of a document each, and the MCP endpoint on the
    session. The API has its own key."""
    monkeypatch.setattr(settings, "mmr_enabled", False)
    monkeypatch.setattr(settings, "api_key", SecretStr(MASTER))
    monkeypatch.setattr("app.services.retrieval_pipeline.get_reranker", lambda: NoopReranker())
    monkeypatch.setattr("app.mcp.server.limiter", RateLimiter())

    collections, documents = {}, {}
    for name in ("a", "b"):
        collection = Collection(name=f"mcp-http-{name}", description="")
        session.add(collection)
        await session.flush()
        document = Document(
            collection_id=collection.id,
            title=f"{name}.md",
            filename=f"{name}.md",
            file_type="md",
            raw_text=f"pgvector in {name}",
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
                text=f"pgvector in {name}",
                token_count=3,
                embedding=_EMBEDDING,
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
    return World(session, collections["a"], collections["b"], documents, endpoint, embedder)


def _text(result) -> str:
    """A tool's error, without the SDK's "Error executing tool <name>: "."""
    assert result.is_error
    return re.sub(r"^Error executing tool \w+: ", "", result.content[0].text)


async def test_a_key_does_not_see_another_collection(world):
    """Audit M-2: a key for one collection gets nothing of another, and
    cannot tell it from one that does not exist."""
    key = await create_key(world.session, "only-a", [world.a])
    missing = 2_000_000_000

    async with world.client(key) as client:
        collections = await client.call_tool("list_collections", {})
        listed = await client.call_tool("list_documents", {})
        of_b = await client.call_tool("list_documents", {"collection_id": world.b})
        search_b = await client.call_tool(
            "search_documents", {"query": "pgvector", "collection_id": world.b}
        )
        search_missing = await client.call_tool(
            "search_documents", {"query": "pgvector", "collection_id": missing}
        )
        search_doc_b = await client.call_tool(
            "search_documents", {"query": "pgvector", "document_id": world.documents["b"]}
        )
        search_any = await client.call_tool("search_documents", {"query": "pgvector"})
        calls = world.embedder.calls
        ask_b = await client.call_tool(
            "ask_documents", {"question": "pgvector?", "collection_id": world.b}
        )
        resources = {str(r.uri) for r in (await client.list_resources()).resources}
        found = await client.read_resource(f"whisperdocs://documents/{world.documents['a']}")
        with pytest.raises(MCPError) as read_b:
            await client.read_resource(f"whisperdocs://documents/{world.documents['b']}")

    assert [c["id"] for c in collections.structured_content["collections"]] == [world.a]
    assert [d["title"] for d in listed.structured_content["documents"]] == ["a.md"]
    assert listed.structured_content["total"] == 1
    assert of_b.structured_content == {"documents": [], "next_cursor": None, "total": 0}
    # Out of reach reads as missing.
    assert _text(search_b) == _text(search_missing).replace(str(missing), str(world.b))
    assert _text(search_b).startswith(f"Collection {world.b} not found")
    assert _text(search_doc_b).startswith(f"Document {world.documents['b']} not found")
    # Without a collection, the key's one.
    assert [c["text"] for c in search_any.structured_content["chunks"]] == ["pgvector in a"]
    assert _text(ask_b).startswith(f"Collection {world.b} not found")
    assert world.embedder.calls == calls
    assert f"whisperdocs://documents/{world.documents['a']}" in resources
    assert f"whisperdocs://documents/{world.documents['b']}" not in resources
    assert found.contents[0].text == "pgvector in a"
    assert f"Document {world.documents['b']} not found" in read_b.value.error.message


async def test_a_key_for_every_collection_or_the_api_key_reaches_both(world):
    every = await create_key(world.session, "every", None)

    for key in (every, MASTER):
        async with world.client(key) as client:
            collections = await client.call_tool("list_collections", {})
            search_b = await client.call_tool(
                "search_documents", {"query": "pgvector", "collection_id": world.b}
            )
            resources = {str(r.uri) for r in (await client.list_resources()).resources}

        ids = {c["id"] for c in collections.structured_content["collections"]}
        assert {world.a, world.b} <= ids
        assert [c["text"] for c in search_b.structured_content["chunks"]] == ["pgvector in b"]
        assert {f"whisperdocs://documents/{d}" for d in world.documents.values()} <= resources


async def test_a_key_for_several_collections_names_one(world):
    key = await create_key(world.session, "a-and-b", [world.b, world.a])

    async with world.client(key) as client:
        unscoped = await client.call_tool("search_documents", {"query": "pgvector"})
        in_b = await client.call_tool(
            "search_documents", {"query": "pgvector", "collection_id": world.b}
        )
        in_doc = await client.call_tool(
            "search_documents", {"query": "pgvector", "document_id": world.documents["a"]}
        )

    a, b = sorted((world.a, world.b))
    assert (
        _text(unscoped) == f"This key reaches several collections ({a}, {b}): pass collection_id."
    )
    assert [c["text"] for c in in_b.structured_content["chunks"]] == ["pgvector in b"]
    assert [c["text"] for c in in_doc.structured_content["chunks"]] == ["pgvector in a"]


async def test_mcp_answers_401_without_a_valid_key(world):
    revoked = await create_key(world.session, "revoked", None)
    await revoke_key(world.session, "revoked")

    none = await world.post({})
    wrong = await world.post({"Authorization": "Bearer wdk_nope"})
    gone = await world.post({"X-API-Key": revoked})

    assert none.status_code == wrong.status_code == gone.status_code == 401
    assert none.headers["www-authenticate"] == "Bearer"
    assert none.json()["code"] == "api_key_required"
    assert wrong.json()["code"] == gone.json()["code"] == "api_key_invalid"


async def test_without_the_api_key_mcp_is_open_as_the_api_is(world, monkeypatch):
    monkeypatch.setattr(settings, "api_key", None)

    async with world.client(None) as client:
        collections = await client.call_tool("list_collections", {})

    assert {world.a, world.b} <= {c["id"] for c in collections.structured_content["collections"]}


async def test_a_host_that_is_not_allowed_gets_nothing(world):
    """The SDK's guard against DNS rebinding, after the key."""
    response = await world.post({"Authorization": f"Bearer {MASTER}", "Host": "evil.example"})

    assert response.status_code == 421


async def test_the_tools_that_pay_count_per_key(world, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
    one = await create_key(world.session, "one", [world.a])
    other = await create_key(world.session, "other", [world.a])
    arguments = {"query": "pgvector"}

    async with world.client(one) as client:
        first = await client.call_tool("search_documents", arguments)
        second = await client.call_tool("ask_documents", {"question": "pgvector?"})
        listed = await client.call_tool("list_collections", {})
    async with world.client(other) as client:
        theirs = await client.call_tool("search_documents", arguments)

    assert not first.is_error
    assert _text(second).startswith("Too many requests: at most 1 a minute. Retry in ")
    assert not listed.is_error
    assert not theirs.is_error


async def test_a_client_of_either_protocol_era_is_scoped_alike(world):
    """Claude Desktop still opens with the `initialize` handshake."""
    key = await create_key(world.session, "eras", [world.a])
    seen = {}
    for mode in ("auto", "legacy"):
        async with world.client(key, mode=mode) as client:
            result = await client.call_tool("list_collections", {})
            seen[client.protocol_version] = [
                c["id"] for c in result.structured_content["collections"]
            ]

    assert seen == {"2026-07-28": [world.a], "2025-11-25": [world.a]}


async def test_the_key_commands(world):
    session = world.session
    key = await create_key(session, "cli", [world.b, world.a, world.a])
    every = await create_key(session, "cli-all", None)

    row = await session.scalar(select(ApiKey).where(ApiKey.name == "cli"))
    assert key.startswith("wdk_") and len(key) > 40
    assert (row.key_hash, row.prefix) == (key_hash(key), key[:12])
    assert key not in (row.key_hash, row.prefix + "x")
    assert row.collection_ids == sorted((world.a, world.b))
    assert await access_for(session, key) == Access("cli", frozenset((world.a, world.b)))
    assert await access_for(session, every) == Access("cli-all", None)
    assert await access_for(session, MASTER) == EVERYTHING

    with pytest.raises(KeyCommandError, match="exists already"):
        await create_key(session, "cli", [world.a])
    with pytest.raises(KeyCommandError, match="No collection 2000000000"):
        await create_key(session, "cli-missing", [world.a, 2_000_000_000])
    with pytest.raises(KeyCommandError, match="needs a collection"):
        await create_key(session, "cli-empty", [])

    await revoke_key(session, "cli")
    assert await access_for(session, key) is None
    with pytest.raises(KeyCommandError, match="not revoked already"):
        await revoke_key(session, "cli")

    listed = {k.name: k for k in await list_keys(session)}
    assert listed["cli"].revoked_at is not None
    assert listed["cli"].revoked_at <= datetime.now(UTC)
    assert listed["cli-all"].collection_ids is None
    assert listed["cli-all"].revoked_at is None
    assert "cli-missing" not in listed
