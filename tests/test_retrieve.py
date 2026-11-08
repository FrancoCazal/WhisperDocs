"""POST /retrieve (audit R20): the chunks for a query, without an answer.

The search itself is tested against Postgres in test_retrieval_db.py; here
the pipeline, the database and the embedder are replaced.
"""

from __future__ import annotations

import httpx
import pytest

from app.config import settings
from app.db import get_db
from app.main import app
from app.services.embedder import EmbeddingError, get_embedder
from app.services.retriever import RetrievedChunk

SECRET = "Incorrect API key provided: sk-..."
EMBEDDING = [0.1, 0.2]


def _chunk(i: int, document_id: int = 7) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=100 + i,
        document_id=document_id,
        document_title="rag.md",
        chunk_index=i,
        text=f"passage {i}",
        relevance=1.0 - i / 10,
        cosine=0.8 if i == 0 else None,
    )


class _Document:
    def __init__(self, id: int, collection_id: int) -> None:
        self.id = id
        self.collection_id = collection_id


class _Db:
    """Documents by id, and the ids of the collections that exist."""

    def __init__(self) -> None:
        self.documents = {7: _Document(7, 3)}
        self.collections = {3, 4}

    async def get(self, _model, id):
        return self.documents.get(id)

    async def scalar(self, statement):
        wanted = statement.whereclause.right.value
        return wanted if wanted in self.collections else None


class _Embedder:
    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail

    async def embed(self, _text):
        if self.fail:
            raise self.fail
        return EMBEDDING


@pytest.fixture
def backend(monkeypatch):
    calls: list[dict] = []

    async def pipeline(**kwargs):
        calls.append(kwargs)
        return [_chunk(0), _chunk(1)]

    async def fake_db():
        yield _Db()

    monkeypatch.setattr("app.api.v1.retrieve.retrieve_pipeline", pipeline)
    app.dependency_overrides[get_db] = fake_db
    app.dependency_overrides[get_embedder] = lambda: _Embedder()
    yield calls
    app.dependency_overrides.clear()


async def _retrieve(body: dict) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
        return await c.post("/retrieve", json=body)


async def test_it_returns_the_chunks_with_their_text_and_no_answer(backend):
    r = await _retrieve({"query": "what is rag", "top_k": 2})

    assert r.status_code == 200
    body = r.json()
    assert "answer" not in body
    assert body["chunks"][0] == {
        "chunk_id": 100,
        "document_id": 7,
        "document_title": "rag.md",
        "chunk_index": 0,
        "text": "passage 0",
        "relevance": 1.0,
        "cosine": 0.8,
    }
    assert body["chunks"][1]["cosine"] is None
    assert body["metadata"]["embedding_model"] == settings.embedding_model
    assert set(body["metadata"]) == {"embedding_model", "reranker", "mmr", "latency_ms"}
    assert backend == [
        {
            "db": backend[0]["db"],
            "query_text": "what is rag",
            "query_embedding": EMBEDDING,
            "collection_id": None,
            "top_k": 2,
            "document_id": None,
        }
    ]


async def test_a_document_scope_searches_its_collection_and_only_it(backend):
    alone = await _retrieve({"query": "q", "document_id": 7})
    with_its_collection = await _retrieve({"query": "q", "document_id": 7, "collection_id": 3})

    assert (alone.status_code, with_its_collection.status_code) == (200, 200)
    assert [(c["collection_id"], c["document_id"]) for c in backend] == [(3, 7), (3, 7)]


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"document_id": 7, "collection_id": 4}, 400, "bad_request"),
        ({"document_id": 8}, 404, "not_found"),
        ({"collection_id": 5}, 404, "not_found"),
    ],
)
async def test_a_scope_that_does_not_hold_is_refused_before_any_call(backend, body, status, code):
    r = await _retrieve({"query": "q", **body})

    assert (r.status_code, r.json()["code"]) == (status, code)
    assert backend == []


@pytest.mark.parametrize(
    "body", [{"query": ""}, {"query": "q", "top_k": 21}, {"query": "q" * 2001}, {}]
)
async def test_an_invalid_body_is_a_422(backend, body):
    r = await _retrieve(body)

    assert (r.status_code, r.json()["code"]) == (422, "validation_error")


async def test_a_failed_embedding_is_a_502_with_its_kind(backend):
    failing = EmbeddingError(f"Failed to generate embeddings: {SECRET}")
    failing.__cause__ = TimeoutError()
    app.dependency_overrides[get_embedder] = lambda: _Embedder(failing)

    r = await _retrieve({"query": "q"})

    assert r.status_code == 502
    assert (r.json()["code"], r.json()["kind"]) == ("upstream_error", "timeout")
    assert SECRET not in r.text
    assert backend == []


async def test_it_counts_against_the_questions_limit(backend, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)

    first = await _retrieve({"query": "q"})
    second = await _retrieve({"query": "q"})

    assert (first.status_code, second.status_code) == (200, 429)
