"""The MCP server (audit M-1) against a real Postgres, through an in-memory MCP
client: only the embedder is replaced.

Needs a database migrated to head (see test_retrieval_db.py); skipped when
TEST_DATABASE_URL is not set. Each test runs inside a transaction that is
rolled back. The database may hold other collections: the tests look for
their own.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import pytest
from mcp import Client
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.config import settings
from app.mcp.server import build_server
from app.models import Chunk, Collection, Document
from app.services.reranker import NoopReranker

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")

_EMBEDDING = [0.0] * 1535 + [1.0]


@pytest.fixture
async def session():
    engine = create_async_engine(DB_URL)
    async with engine.connect() as conn:
        outer = await conn.begin()
        async with AsyncSession(bind=conn, expire_on_commit=False) as s:
            yield s
        await outer.rollback()
    await engine.dispose()


class _Embedder:
    async def embed(self, _text):
        return _EMBEDDING


@pytest.fixture
async def seeded(session, monkeypatch):
    """A collection of two documents, and a server on the session."""
    monkeypatch.setattr(settings, "mmr_enabled", False)
    monkeypatch.setattr("app.services.retrieval_pipeline.get_reranker", lambda: NoopReranker())

    collection = Collection(name="mcp-test", description="for the MCP tests")
    session.add(collection)
    await session.flush()
    documents = {}
    for title, texts in {
        "mine.md": ["pgvector basics", "pgvector indexes"],
        "other.md": ["pgvector internals"],
    }.items():
        document = Document(
            collection_id=collection.id,
            title=title,
            filename=title,
            file_type="md",
            raw_text="\n\n".join(texts),
            chunk_count=len(texts),
            total_tokens=2 * len(texts),
            language="en",
        )
        session.add(document)
        await session.flush()
        session.add_all(
            Chunk(
                document_id=document.id,
                chunk_index=i,
                text=text,
                token_count=2,
                embedding=_EMBEDDING,
                language="en",
            )
            for i, text in enumerate(texts)
        )
        documents[title] = document.id
    await session.flush()

    @asynccontextmanager
    async def sessions():
        yield session

    server = build_server(sessions=sessions, embedder=lambda: _Embedder())
    return server, collection.id, documents


async def test_the_lists_give_the_ids_that_scope_a_search(seeded):
    server, collection_id, documents = seeded

    async with Client(server) as client:
        collections = await client.call_tool("list_collections", {})
        page = await client.call_tool("list_documents", {"collection_id": collection_id})

    ours = [c for c in collections.structured_content["collections"] if c["id"] == collection_id]
    assert ours == [
        {
            "id": collection_id,
            "name": "mcp-test",
            "description": "for the MCP tests",
            "created_at": ours[0]["created_at"],
            "document_count": 2,
            "chunk_count": 3,
            "total_tokens": 6,
        }
    ]
    listed = page.structured_content
    assert [(d["title"], d["uri"]) for d in listed["documents"]] == [
        ("other.md", f"whisperdocs://documents/{documents['other.md']}"),
        ("mine.md", f"whisperdocs://documents/{documents['mine.md']}"),
    ]
    assert (listed["total"], listed["next_cursor"]) == (2, None)


async def test_a_page_of_documents_points_to_the_next(seeded):
    server, collection_id, documents = seeded

    async with Client(server) as client:
        first = await client.call_tool(
            "list_documents", {"collection_id": collection_id, "limit": 1}
        )
        cursor = first.structured_content["next_cursor"]
        second = await client.call_tool(
            "list_documents", {"collection_id": collection_id, "limit": 1, "cursor": cursor}
        )

    assert [d["id"] for d in first.structured_content["documents"]] == [documents["other.md"]]
    assert [d["id"] for d in second.structured_content["documents"]] == [documents["mine.md"]]
    assert second.structured_content["next_cursor"] is None


async def test_search_documents_runs_the_whole_search(seeded):
    server, collection_id, documents = seeded

    async with Client(server) as client:
        in_collection = await client.call_tool(
            "search_documents", {"query": "pgvector", "collection_id": collection_id}
        )
        in_document = await client.call_tool(
            "search_documents", {"query": "pgvector", "document_id": documents["mine.md"]}
        )

    assert sorted(
        (c["document_title"], c["text"]) for c in in_collection.structured_content["chunks"]
    ) == [
        ("mine.md", "pgvector basics"),
        ("mine.md", "pgvector indexes"),
        ("other.md", "pgvector internals"),
    ]
    assert {c["document_title"] for c in in_document.structured_content["chunks"]} == {"mine.md"}


async def test_each_document_is_a_resource_with_its_full_text(seeded):
    server, _collection_id, documents = seeded
    uri = f"whisperdocs://documents/{documents['mine.md']}"

    async with Client(server) as client:
        resources = {str(r.uri): r for r in (await client.list_resources()).resources}
        text = (await client.read_resource(uri)).contents[0].text

    assert resources[uri].title == "mine.md"
    assert resources[uri].mime_type == "text/plain"
    assert "'mcp-test'" in resources[uri].description
    assert resources[uri].size == len(text.encode())
    assert text == "pgvector basics\n\npgvector indexes"
