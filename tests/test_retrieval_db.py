"""Retrieval against a real Postgres: full-text leg (audit R1, N3) and the
collection filter on the HNSW index (R3).

Needs a database migrated to head: point TEST_DATABASE_URL at it, e.g.
`postgresql+asyncpg://whisperdocs:pw@localhost:5432/whisperdocs_test`, and run
`alembic upgrade head` there first. Skipped when the variable is not set, so
the offline suite is unaffected. Each test runs inside a transaction that is
rolled back.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.models import Chunk, Collection, Document
from app.services.retriever import VectorRetriever

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


async def _collection(session, name: str, docs: dict[str, tuple[str, list[str]]]) -> int:
    """docs: title -> (language, [chunk texts])."""
    coll = Collection(name=name)
    session.add(coll)
    await session.flush()
    for title, (language, texts) in docs.items():
        doc = Document(
            collection_id=coll.id,
            title=title,
            filename=title,
            file_type="md",
            raw_text="\n".join(texts),
            language=language,
        )
        session.add(doc)
        await session.flush()
        session.add_all(
            Chunk(
                document_id=doc.id,
                chunk_index=i,
                text=t,
                token_count=len(t.split()),
                embedding=_EMBEDDING,
                language=language,
            )
            for i, t in enumerate(texts)
        )
    await session.flush()
    return coll.id


async def _titles(session, question: str, collection_id: int) -> list[str]:
    chunks = await VectorRetriever(session).lexical_retrieve(question, collection_id=collection_id)
    return [c.document_title for c in chunks]


async def test_natural_language_question_matches(session):
    cid = await _collection(
        session,
        "fts-nl",
        {
            "indexes.md": (
                "en",
                ["HNSW indexes trade memory for recall; IVFFlat clusters vectors."],
            ),
            "other.md": ("en", ["Celery runs background jobs outside the web process."]),
        },
    )

    # As shipped (AND over every word) this matched nothing: no chunk says
    # "how", "does" or "to".
    titles = await _titles(session, "How does HNSW compare to IVFFlat?", cid)

    assert titles == ["indexes.md"]


async def test_more_shared_terms_rank_higher(session):
    cid = await _collection(
        session,
        "fts-rank",
        {
            "both.md": ("en", ["HNSW and IVFFlat are the two pgvector index types."]),
            "one.md": ("en", ["IVFFlat needs a training step before it can be built."]),
        },
    )

    titles = await _titles(session, "Compare HNSW with IVFFlat", cid)

    assert titles == ["both.md", "one.md"]


async def test_minus_and_quotes_are_not_operators(session):
    cid = await _collection(
        session,
        "fts-n3",
        {"planner.md": ("en", ["Rows excluded by the planner never reach the executor."])},
    )

    # websearch_to_tsquery read "-excluded" as NOT excluded.
    titles = await _titles(session, 'Why are "rows" -excluded?', cid)

    assert titles == ["planner.md"]


async def test_spanish_matches_without_accents_and_by_stem(session):
    cid = await _collection(
        session,
        "fts-es",
        {"indices.md": ("es", ["El índice HNSW acelera las búsquedas vectoriales."])},
    )

    titles = await _titles(session, "indice para busqueda vectorial", cid)

    assert titles == ["indices.md"]


async def test_stopword_only_question_returns_nothing(session):
    cid = await _collection(
        session, "fts-stop", {"a.md": ("en", ["How it is done is what it is."])}
    )

    assert await _titles(session, "how is it", cid) == []


async def test_other_collections_are_not_searched(session):
    mine = await _collection(session, "fts-mine", {"mine.md": ("en", ["pgvector basics"])})
    await _collection(session, "fts-theirs", {"theirs.md": ("en", ["pgvector internals"])})

    assert await _titles(session, "pgvector", mine) == ["mine.md"]


def _unit(axis: int, noise_axis: int | None = None, noise: float = 0.0) -> list[float]:
    v = [0.0] * 1536
    v[axis] = 1.0
    if noise_axis is not None:
        v[noise_axis] = noise
    return v


async def test_small_collection_is_not_starved_by_the_hnsw_scan(session):
    """A 3-chunk collection next to a 400-chunk one, every big chunk closer to
    the query. The HNSW scan visits `ef_search` (40) candidates and filters
    after, so without iterative scan the small collection came back empty.
    Iterative scan alone is not enough here either: the 400 near-duplicate
    vectors leave part of the graph unreachable, and the scan ran out after
    95 to 350 rows. The exact fallback is what returns the small collection."""
    big = Collection(name="hnsw-big")
    small = Collection(name="hnsw-small")
    session.add_all([big, small])
    await session.flush()
    docs = {}
    for coll in (big, small):
        doc = Document(
            collection_id=coll.id,
            title=f"{coll.name}.md",
            filename=f"{coll.name}.md",
            file_type="md",
            raw_text="x",
            language="en",
        )
        session.add(doc)
        docs[coll.name] = doc
    await session.flush()
    session.add_all(
        Chunk(
            document_id=docs["hnsw-big"].id,
            chunk_index=i,
            text=f"big {i}",
            token_count=2,
            embedding=_unit(0, noise_axis=1 + i, noise=0.05),
            language="en",
        )
        for i in range(400)
    )
    session.add_all(
        Chunk(
            document_id=docs["hnsw-small"].id,
            chunk_index=i,
            text=f"small {i}",
            token_count=2,
            embedding=_unit(1000 + i),
            language="en",
        )
        for i in range(3)
    )
    await session.flush()
    # Force the plan Postgres picks on large tables: walk the HNSW index in
    # distance order and filter by collection on the way.
    for setting in ("enable_seqscan", "enable_sort", "enable_hashjoin"):
        await session.execute(text(f"SET LOCAL {setting} = off"))

    chunks = await VectorRetriever(session).retrieve(_unit(0), collection_id=small.id, top_k=3)

    assert sorted(c.text for c in chunks) == ["small 0", "small 1", "small 2"]


async def test_exact_order_does_not_use_the_hnsw_index(session):
    """The fallback relies on `distance + 0` hiding the operator from the
    planner. Check the plan, so a planner change cannot silently undo it."""
    distance = Chunk.embedding.cosine_distance(_unit(0))
    for setting in ("enable_seqscan", "enable_sort"):
        await session.execute(text(f"SET LOCAL {setting} = off"))

    async def explain(order) -> str:
        stmt = select(Chunk.id).order_by(order).limit(3)
        sql = stmt.compile(dialect=session.bind.dialect, compile_kwargs={"literal_binds": True})
        rows = await session.execute(text(f"EXPLAIN {sql}"))
        return "\n".join(rows.scalars())

    assert "idx_chunk_embedding" in await explain(distance)
    assert "idx_chunk_embedding" not in await explain(distance + 0)
