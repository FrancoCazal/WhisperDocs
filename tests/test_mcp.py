"""The MCP server (audit M-1), through an in-memory MCP client.

The database, the embedder, the retrieval pipeline and the model are replaced:
the search itself is tested against Postgres in test_mcp_db.py. A tool and its
API route call the same service, and the tests here check they answer alike.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from mcp import Client, MCPError

from app.db import get_db
from app.main import app
from app.mcp.server import build_server
from app.models import QueryLog
from app.services.embedder import EmbeddingError, get_embedder
from app.services.llm.base import LLMResponse
from app.services.retriever import RetrievedChunk

ROOT = Path(__file__).resolve().parent.parent
SECRET = "Incorrect API key provided: sk-..."
EMBEDDING = [0.1, 0.2]
READS = {
    "search_documents",
    "ask_documents",
    "extract_fields",
    "list_collections",
    "list_documents",
}
WRITES = {"ingest_document", "delete_document", "move_document"}
TOOLS = READS | WRITES


def _chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=100 + i,
        document_id=7,
        document_title="rag.md",
        chunk_index=i,
        text=f"passage {i}",
        relevance=1.0 - i / 10,
        cosine=0.8 if i == 0 else None,
    )


class _Document:
    def __init__(self, id: int, collection_id: int, raw_text: str = "") -> None:
        self.id = id
        self.collection_id = collection_id
        self.raw_text = raw_text


class _Db:
    """Documents by id, the ids of the collections that exist, and what was
    added (the query logs)."""

    def __init__(self) -> None:
        self.documents = {7: _Document(7, 3, raw_text="# RAG\n\nThe whole text.")}
        self.collections = {3, 4}
        self.added: list[object] = []

    async def get(self, _model, id):
        return self.documents.get(id)

    async def scalar(self, statement):
        wanted = statement.whereclause.right.value
        return wanted if wanted in self.collections else None

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        pass


class _Embedder:
    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.calls = 0

    async def embed(self, _text):
        self.calls += 1
        if self.fail:
            raise self.fail
        return EMBEDDING


class _Provider:
    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail

    async def generate(self, **_kwargs) -> LLMResponse:
        if self.fail:
            raise self.fail
        return LLMResponse(
            text="RAG retrieves first [1], then generates [2].",
            tokens_input=10,
            tokens_output=5,
            model="fake-model",
            latency_ms=1,
        )


class Backend:
    def __init__(self) -> None:
        self.db = _Db()
        self.embedder = _Embedder()
        self.provider = _Provider()
        self.searches: list[dict] = []

    def server(self):
        @asynccontextmanager
        async def session():
            yield self.db

        return build_server(sessions=session, embedder=lambda: self.embedder)


@pytest.fixture
def backend(monkeypatch):
    b = Backend()

    async def pipeline(**kwargs):
        b.searches.append({k: v for k, v in kwargs.items() if k != "db"})
        return [_chunk(0), _chunk(1)]

    async def fake_db():
        yield b.db

    monkeypatch.setattr("app.services.search.retrieve_pipeline", pipeline)
    monkeypatch.setattr("app.services.answering.retrieve_pipeline", pipeline)
    monkeypatch.setattr(
        "app.services.answering.llm_service_for", lambda _role: (b.provider, "fake-model", 0.0)
    )
    app.dependency_overrides[get_db] = fake_db
    app.dependency_overrides[get_embedder] = lambda: b.embedder
    yield b
    app.dependency_overrides.clear()


async def _api(path: str, body: dict) -> dict:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
        r = await c.post(path, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _without_latency(result: dict) -> dict:
    return {**result, "metadata": {**result["metadata"], "latency_ms": 0}}


async def test_the_tools_are_few_typed_and_say_what_they_change(backend):
    async with Client(backend.server()) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}

    assert set(tools) == TOOLS
    for tool in tools.values():
        assert tool.output_schema is not None, tool.name
        assert tool.description and tool.title
        assert tool.annotations.open_world_hint is False
    assert {n for n, t in tools.items() if t.annotations.read_only_hint} == READS
    # Deleting is the one change that cannot be undone (BE-36).
    assert [n for n, t in tools.items() if t.annotations.destructive_hint] == ["delete_document"]
    # ask_documents and extract_fields pay a model call and differ each time;
    # ingest_document adds a document, then refuses the same one.
    assert sorted(n for n, t in tools.items() if not t.annotations.idempotent_hint) == [
        "ask_documents",
        "extract_fields",
        "ingest_document",
    ]
    for name in ("search_documents", "ask_documents"):
        assert tools[name].input_schema["properties"]["top_k"]["maximum"] == 20
    assert tools["list_documents"].input_schema["properties"]["limit"]["maximum"] == 100
    # The confirmation is the user's to give, not an argument the model fills.
    for name in ("delete_document", "move_document"):
        assert set(tools[name].input_schema["properties"]) <= {
            "document_id",
            "target_collection_id",
        }


async def test_search_documents_returns_what_post_retrieve_returns(backend):
    request = {"query": "what is rag", "collection_id": 3, "top_k": 2}

    async with Client(backend.server()) as client:
        result = await client.call_tool("search_documents", request)
    api = await _api("/retrieve", request)

    assert not result.is_error
    assert _without_latency(result.structured_content) == _without_latency(api)
    assert result.structured_content["chunks"][0]["text"] == "passage 0"
    assert (
        backend.searches[0]
        == backend.searches[1]
        == {
            "query_text": "what is rag",
            "query_embedding": EMBEDDING,
            "collection_id": 3,
            "top_k": 2,
            "document_id": None,
        }
    )


async def test_a_document_scope_searches_its_own_collection(backend):
    async with Client(backend.server()) as client:
        result = await client.call_tool("search_documents", {"query": "q", "document_id": 7})

    assert not result.is_error
    assert (backend.searches[0]["collection_id"], backend.searches[0]["document_id"]) == (3, 7)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"document_id": 8}, "Document 8 not found"),
        ({"collection_id": 5}, "Collection 5 not found"),
        ({"document_id": 7, "collection_id": 4}, "Document 7 is not in collection 4"),
    ],
)
async def test_a_scope_that_does_not_hold_is_a_tool_error_before_any_call(
    backend, arguments, message
):
    async with Client(backend.server()) as client:
        result = await client.call_tool("search_documents", {"query": "q", **arguments})

    assert result.is_error
    assert message in result.content[0].text
    assert "list_collections" in result.content[0].text
    assert (backend.embedder.calls, backend.searches) == (0, [])


@pytest.mark.parametrize(
    "arguments", [{"query": ""}, {"query": "q", "top_k": 21}, {"query": "q" * 2001}, {}]
)
async def test_arguments_out_of_bounds_are_refused_before_any_call(backend, arguments):
    async with Client(backend.server()) as client:
        result = await client.call_tool("search_documents", arguments)

    assert result.is_error
    assert (backend.embedder.calls, backend.searches) == (0, [])


async def test_a_failed_embedding_says_why_and_not_the_providers_text(backend):
    failing = EmbeddingError(f"Failed to generate embeddings: {SECRET}")
    failing.__cause__ = TimeoutError()
    backend.embedder.fail = failing

    async with Client(backend.server()) as client:
        result = await client.call_tool("search_documents", {"query": "q"})

    assert result.is_error
    assert "Embedding the query failed: it timed out." in result.content[0].text
    assert SECRET not in result.content[0].text


async def test_no_embedder_is_a_tool_error_not_a_crash(backend):
    def no_embedder():
        raise EmbeddingError("OPENAI_API_KEY not configured")

    @asynccontextmanager
    async def session():
        yield backend.db

    async with Client(build_server(sessions=session, embedder=no_embedder)) as client:
        result = await client.call_tool("search_documents", {"query": "q"})

    assert result.is_error
    assert result.content[0].text.endswith(
        "Embedding the query failed: the server's log has the details."
    )


async def test_ask_documents_answers_like_post_query_and_logs_it(backend):
    request = {"question": "what is rag", "collection_id": 3, "top_k": 2}

    async with Client(backend.server()) as client:
        result = await client.call_tool("ask_documents", request)
    api = await _api("/query", request)

    assert not result.is_error
    assert _without_latency(result.structured_content) == _without_latency(api)
    assert result.structured_content["answer"] == "RAG retrieves first [1], then generates [2]."
    assert [c["chunk_id"] for c in result.structured_content["citations"]] == [100, 101]
    logs = [o for o in backend.db.added if isinstance(o, QueryLog)]
    assert [(log.question, log.collection_id) for log in logs] == [("what is rag", 3)] * 2


async def test_ask_documents_refuses_a_missing_collection_before_any_call(backend):
    async with Client(backend.server()) as client:
        result = await client.call_tool("ask_documents", {"question": "q", "collection_id": 5})

    assert result.is_error
    assert "Collection 5 not found" in result.content[0].text
    assert (backend.embedder.calls, backend.searches, backend.db.added) == (0, [], [])


async def test_a_failed_generation_says_why_and_not_the_providers_text(backend):
    backend.provider.fail = ConnectionError(SECRET)

    async with Client(backend.server()) as client:
        result = await client.call_tool("ask_documents", {"question": "q"})

    assert result.is_error
    assert "Generation failed: the provider could not be reached." in result.content[0].text
    assert SECRET not in result.content[0].text
    assert backend.db.added == []


async def test_a_document_resource_is_its_full_text(backend):
    async with Client(backend.server()) as client:
        templates = (await client.list_resource_templates()).resource_templates
        found = await client.read_resource("whisperdocs://documents/7")
        with pytest.raises(MCPError) as missing:
            await client.read_resource("whisperdocs://documents/8")

    assert [t.uri_template for t in templates] == ["whisperdocs://documents/{document_id}"]
    assert found.contents[0].text == "# RAG\n\nThe whole text."
    assert found.contents[0].mime_type == "text/plain"
    # Invalid params: the 2026-07-28 code for a resource that is not there.
    assert missing.value.error.code == -32602
    assert "Document 8 not found" in missing.value.error.message


async def test_a_client_of_either_protocol_era_gets_the_same_server(backend):
    """Claude Desktop and other clients still open with the `initialize`
    handshake (up to 2025-11-25); the SDK serves both eras."""
    results = {}
    for mode in ("auto", "legacy"):
        async with Client(backend.server(), mode=mode) as client:
            tools = sorted(t.name for t in (await client.list_tools()).tools)
            result = await client.call_tool("search_documents", {"query": "q"})
            results[client.protocol_version] = (tools, result.structured_content["chunks"])

    assert set(results) == {"2026-07-28", "2025-11-25"}
    assert results["2026-07-28"] == results["2025-11-25"]


def test_the_stdio_server_writes_only_protocol_messages_to_stdout(tmp_path):
    """`python -m app.mcp` as Claude Desktop starts it, with the `initialize`
    handshake. Both calls fail, on a database and an embedding API that are
    not there (port 9: nothing is sent anywhere), and the server logs each
    failure. The spec allows only MCP messages on stdout: a log line there
    breaks clients that read every line as JSON-RPC. The SDK points the
    process's stdout at stderr while it serves; anything written before, or
    around it, fails here."""
    env = {
        **os.environ,
        "DATABASE_URL": "postgresql+asyncpg://test:test@127.0.0.1:9/test",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "http://127.0.0.1:9",
        "LOG_JSON": "1",
    }
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "search_documents", "arguments": {"query": "q"}},
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "list_collections", "arguments": {}},
        },
    ]
    stderr_path = tmp_path / "stderr.log"
    with stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-m", "app.mcp"],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        assert process.stdin and process.stdout
        # Every line the server writes, read off the pipe as it comes; None at
        # the end. A thread, so that a server that hangs fails the test.
        lines: queue.Queue[bytes | None] = queue.Queue()

        def read_stdout() -> None:
            assert process.stdout
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        threading.Thread(target=read_stdout, daemon=True).start()
        written: list[dict] = []
        try:
            process.stdin.write(b"".join(json.dumps(r).encode() + b"\n" for r in requests))
            process.stdin.flush()
            while {m.get("id") for m in written} < {1, 2, 3}:
                line = lines.get(timeout=60)
                assert line is not None, f"the server closed stdout after {written}"
                written.append(json.loads(line))
            process.stdin.close()
            while (line := lines.get(timeout=30)) is not None:
                written.append(json.loads(line))
            process.wait(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    assert {m["jsonrpc"] for m in written} == {"2.0"}
    answered = {m["id"]: m for m in written if "id" in m}
    assert answered[1]["result"]["protocolVersion"] == "2025-11-25"
    search, collections = answered[2]["result"], answered[3]["result"]
    assert search["isError"] and collections["isError"]
    assert search["content"][0]["text"].endswith(
        "Embedding the query failed: the provider could not be reached."
    )
    logged = stderr_path.read_text(encoding="utf-8", errors="replace")
    assert "mcp_search_embedding_failed" in logged
