"""Each chunk is stored, and each text embedded once per model (audit R12):
the vectors a chunk's text already has are reused, and a text the document
repeats is embedded once. A duplicate document's error reaches the client."""

from __future__ import annotations

import hashlib

from app.services import ingestion
from app.services.ingestion import (
    DuplicateDocumentError,
    FreshChunk,
    embed_chunks,
    job_error_message,
    reusable_embeddings,
)


class _Embedder:
    model_id = "fake-embedding"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text))] for text in texts]


def _chunk(index: int, text: str) -> FreshChunk:
    return FreshChunk(index, text, 1, f"hash-{text}")


async def test_a_text_with_a_stored_vector_is_not_embedded_again():
    chunks = [_chunk(0, "new"), _chunk(1, "known"), _chunk(2, "other")]
    embedder = _Embedder()

    vectors = await embed_chunks(chunks, embedder, {"hash-known": [9.0]})  # type: ignore[arg-type]

    assert embedder.calls == [["new", "other"]]
    assert vectors == [[3.0], [9.0], [5.0]]


async def test_a_text_the_document_repeats_is_embedded_once_and_stored_each_time():
    chunks = [_chunk(0, "footer"), _chunk(1, "body"), _chunk(2, "footer")]
    embedder = _Embedder()

    vectors = await embed_chunks(chunks, embedder, {})  # type: ignore[arg-type]

    assert embedder.calls == [["footer", "body"]]
    assert vectors == [[6.0], [4.0], [6.0]]


async def test_batches_report_how_many_chunks_have_a_vector():
    chunks = [_chunk(i, f"text {i}") for i in range(5)] + [_chunk(5, "known")]
    embedder = _Embedder()
    seen: list[int] = []

    async def progress(done: int) -> None:
        seen.append(done)

    await embed_chunks(
        chunks,
        embedder,  # type: ignore[arg-type]
        {"hash-known": [1.0]},
        batch_size=2,
        on_batch=progress,
    )

    assert [len(call) for call in embedder.calls] == [2, 2, 1]
    # The reused chunk counts from the start.
    assert seen == [3, 5, 6]


async def test_a_document_whose_every_text_is_known_calls_nothing():
    chunks = [_chunk(0, "a"), _chunk(1, "b")]
    embedder = _Embedder()

    vectors = await embed_chunks(
        chunks,
        embedder,  # type: ignore[arg-type]
        {"hash-a": [1.0], "hash-b": [2.0]},
    )

    assert embedder.calls == [] and vectors == [[1.0], [2.0]]


async def test_no_hashes_ask_the_database_nothing():
    class _NoDb:
        async def execute(self, *_a, **_k):
            raise AssertionError("queried")

    assert await reusable_embeddings(_NoDb(), [], "m") == {}


def test_a_duplicate_names_the_document_the_collection_holds():
    error = DuplicateDocumentError(12, "guide.pdf")

    assert str(error).startswith(
        'The collection already holds this document as "guide.pdf" (id 12).'
    )
    assert error.document_id == 12
    assert "already holds this document." in str(DuplicateDocumentError())


def test_a_failed_job_says_its_document_is_a_duplicate():
    assert job_error_message(DuplicateDocumentError(3, "a.md")) == str(
        DuplicateDocumentError(3, "a.md")
    )


def test_the_content_hash_is_the_sha256_of_the_utf8_text():
    # SHA-256 of nothing; the database test checks Postgres computes the same.
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    assert ingestion.content_hash("") == empty
    assert ingestion.content_hash("año") == hashlib.sha256("año".encode()).hexdigest()
    assert ingestion.content_hash("año") != ingestion.content_hash("ano")
