from dataclasses import dataclass, replace

from sqlalchemy import bindparam, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Chunk, Document
from app.services.fulltext import or_tsquery_sql


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval.

    `relevance` is the score the list is ordered by, between 0 and 1:
      - vector leg (`retrieve`): cosine similarity;
      - full-text leg (`lexical_retrieve`): rank divided by the best rank;
      - hybrid (`hybrid_retrieve`): RRF score divided by the best score
        possible, so 1.0 means first in every leg that returned rows;
      - after the cross-encoder: its score, min-max normalized.
    `cosine` is the cosine similarity to the query, or None when only the
    full-text leg returned the chunk.
    """

    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    text: str
    relevance: float
    cosine: float | None = None


def fuse_rrf(
    vector: list[RetrievedChunk],
    lexical: list[RetrievedChunk],
    top_k: int,
    k_rrf: int = 60,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion of two ranked lists.

    score(c) = sum over legs of 1 / (k_rrf + rank of c in that leg). The result
    is ordered by that score; `relevance` is the score divided by the best one
    possible (rank 1 in every leg that returned rows). Chunks keep the cosine
    they got in the vector leg.
    """
    scores: dict[int, float] = {}
    by_id: dict[int, RetrievedChunk] = {}
    for leg in (vector, lexical):
        for rank, chunk in enumerate(leg, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k_rrf + rank)
            by_id.setdefault(chunk.chunk_id, chunk)  # vector leg first: keeps cosine

    legs_with_rows = sum(1 for leg in (vector, lexical) if leg)
    if not legs_with_rows:
        return []
    best_possible = legs_with_rows / (k_rrf + 1)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [replace(by_id[cid], relevance=round(score / best_possible, 4)) for cid, score in ranked]


class VectorRetriever:
    """Hybrid retrieval: pgvector cosine + Postgres full-text, fused with RRF."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def retrieve(
        self,
        query_embedding: list[float],
        collection_id: int | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Vector leg: nearest chunks by cosine distance.

        On a large table Postgres walks the HNSW index in distance order and
        applies the collection filter afterwards, over only `hnsw.ef_search`
        candidates (40). A collection that is a small share of the corpus then
        comes back short, or empty. Iterative scan (pgvector >= 0.8) keeps
        walking until LIMIT rows pass the filter. `relaxed_order` may return
        rows slightly out of order, so they are re-sorted below. SET LOCAL
        lasts until the session's transaction ends.
        """
        await self.db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")

        query = (
            select(Chunk, Document, distance)
            .join(Document, Chunk.document_id == Document.id)
            .options(selectinload(Chunk.document))
            .order_by(distance)
            .limit(top_k)
        )

        if collection_id is not None:
            query = query.where(Document.collection_id == collection_id)

        result = await self.db.execute(query)
        rows = sorted(result.all(), key=lambda row: row[2])

        chunks: list[RetrievedChunk] = []
        for chunk, document, dist in rows:
            cosine = round(1.0 - float(dist), 4)
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    document_title=document.title,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    relevance=cosine,
                    cosine=cosine,
                )
            )
        return chunks

    async def lexical_retrieve(
        self,
        query_text: str,
        collection_id: int | None = None,
        limit: int = 20,
    ) -> list[RetrievedChunk]:
        """Full-text leg: chunks sharing any lexeme with the question.

        Ranked by `ts_rank_cd`, Postgres cover-density ranking. It is not
        BM25: there is no IDF term. No length normalization: on the golden set
        it gave the best hybrid precision and recall of the variants tried
        (`scripts/retrieval_bench.py`; see docs/adr/0003).
        """
        if not query_text or not query_text.strip():
            return []

        col_filter = "AND d.collection_id = :col_id" if collection_id is not None else ""
        sql = f"""
            SELECT
                c.id AS chunk_id,
                d.id AS document_id,
                d.title AS document_title,
                c.chunk_index AS chunk_index,
                c.text AS text,
                ts_rank_cd(c.tsv, q.tsq) AS rank
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            CROSS JOIN (SELECT {or_tsquery_sql("q")} AS tsq) q
            WHERE c.tsv @@ q.tsq
              {col_filter}
            ORDER BY rank DESC, c.id
            LIMIT :limit
        """
        stmt = text(sql).bindparams(
            bindparam("q", value=query_text),
            bindparam("limit", value=limit),
        )
        if collection_id is not None:
            stmt = stmt.bindparams(bindparam("col_id", value=collection_id))
        rows = list((await self.db.execute(stmt)).mappings())
        if not rows:
            return []

        best = float(rows[0]["rank"]) or 1.0
        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                document_title=row["document_title"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                relevance=round(float(row["rank"]) / best, 4),
            )
            for row in rows
        ]

    async def hybrid_retrieve(
        self,
        query_text: str,
        query_embedding: list[float],
        collection_id: int | None = None,
        top_k: int = 5,
        k_rrf: int = 60,
        candidate_pool: int = 20,
    ) -> list[RetrievedChunk]:
        """Top `candidate_pool` from each leg, fused with RRF, cut to `top_k`.

        When the full-text leg returns nothing, the result is the vector leg.
        """
        vector = await self.retrieve(
            query_embedding=query_embedding,
            collection_id=collection_id,
            top_k=candidate_pool,
        )
        lexical = await self.lexical_retrieve(
            query_text, collection_id=collection_id, limit=candidate_pool
        )
        return fuse_rrf(vector, lexical, top_k=top_k, k_rrf=k_rrf)
