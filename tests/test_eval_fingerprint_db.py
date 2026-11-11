"""The fingerprint of an eval run's collection against a real Postgres: the
same chunk texts give the same hash whatever their ids, and anything else
moves it. A diff warns when two runs searched different chunks.

Needs TEST_DATABASE_URL, a database migrated to head; each test runs inside a
transaction that is rolled back.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.config import settings
from app.eval.protocol import collection_fingerprint
from app.models import Chunk, Collection, Document

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")

_EMBEDDING = [0.0] * (settings.embedding_dimension - 1) + [1.0]


@pytest.fixture
async def session():
    engine = create_async_engine(DB_URL)
    async with engine.connect() as conn:
        outer = await conn.begin()
        async with AsyncSession(bind=conn, expire_on_commit=False) as s:
            yield s
        await outer.rollback()
    await engine.dispose()


async def _collection(session, texts: list[str], text_hash: bool = True) -> int:
    # Collection names are unique.
    coll = Collection(name=f"fingerprint {uuid.uuid4()}")
    session.add(coll)
    await session.flush()
    doc = Document(
        collection_id=coll.id,
        title="doc.md",
        filename="doc.md",
        file_type="md",
        raw_text="\n".join(texts),
    )
    session.add(doc)
    await session.flush()
    session.add_all(
        Chunk(
            document_id=doc.id,
            chunk_index=i,
            text=text,
            token_count=1,
            text_hash=f"{i}-{text}" if text_hash else None,
            embedding=_EMBEDDING,
        )
        for i, text in enumerate(texts)
    )
    await session.flush()
    return coll.id


async def test_the_same_chunks_give_the_same_fingerprint_whatever_their_ids(session):
    first = await _collection(session, ["alpha", "beta"])
    # Seeded again: new ids, the same texts.
    again = await _collection(session, ["alpha", "beta"])

    assert await collection_fingerprint(session, first) == await collection_fingerprint(
        session, again
    )
    count, digest = await collection_fingerprint(session, first)
    assert count == 2 and len(digest) == 12


async def test_another_chunking_moves_the_fingerprint(session):
    before = await collection_fingerprint(session, await _collection(session, ["alpha", "beta"]))
    rechunked = await collection_fingerprint(session, await _collection(session, ["alpha beta"]))
    grown = await collection_fingerprint(
        session, await _collection(session, ["alpha", "beta", "gamma"])
    )

    assert len({before[1], rechunked[1], grown[1]}) == 3
    assert (rechunked[0], grown[0]) == (1, 3)


async def test_a_chunk_without_a_text_hash_is_hashed_by_its_text(session):
    one = await collection_fingerprint(session, await _collection(session, ["a"], text_hash=False))
    other = await collection_fingerprint(
        session, await _collection(session, ["b"], text_hash=False)
    )

    assert one[0] == other[0] == 1
    assert one[1] != other[1]


async def test_an_empty_collection_has_a_fingerprint_too(session):
    coll = Collection(name="fingerprint empty")
    session.add(coll)
    await session.flush()

    count, digest = await collection_fingerprint(session, coll.id)

    assert count == 0 and len(digest) == 12
