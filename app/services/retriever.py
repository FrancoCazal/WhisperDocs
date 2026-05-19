from dataclasses import dataclass

from sqlalchemy import bindparam, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Chunk, Document


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    text: str
    relevance: float


class VectorRetriever:
    """Cosine similarity + BM25 hybrid retrieval over pgvector + tsvector."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def retrieve(
        self,
        query_embedding: list[float],
        collection_id: int | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Pure vector retrieval. Kept for back-compat / pure-semantic queries."""
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
        rows = result.all()

        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=document.id,
                document_title=document.title,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                relevance=round(1.0 - float(dist), 4),
            )
            for chunk, document, dist in rows
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
        """Reciprocal Rank Fusion of vector cosine + BM25 (ts_rank_cd).

        Pulls top `candidate_pool` from each strategy, fuses by RRF score
        (sum of 1 / (k_rrf + rank)), returns top_k unique chunks.

        BM25 query uses websearch_to_tsquery for natural-language input.
        Falls back to vector-only if query_text is empty or yields zero
        BM25 hits.
        """
        # Vector candidates
        vec_result = await self.retrieve(
            query_embedding=query_embedding,
            collection_id=collection_id,
            top_k=candidate_pool,
        )
        vec_ranks: dict[int, tuple[RetrievedChunk, int]] = {
            c.chunk_id: (c, i + 1) for i, c in enumerate(vec_result)
        }

        # BM25 candidates
        bm25_rows: list = []
        if query_text and query_text.strip():
            bm25_sql = """
                SELECT
                    c.id AS chunk_id,
                    d.id AS document_id,
                    d.title AS document_title,
                    c.chunk_index AS chunk_index,
                    c.text AS text,
                    ts_rank_cd(c.tsv, websearch_to_tsquery('simple', :q)) AS rank
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.tsv @@ websearch_to_tsquery('simple', :q)
                  {col_filter}
                ORDER BY rank DESC
                LIMIT :limit
            """
            col_filter = "AND d.collection_id = :col_id" if collection_id is not None else ""
            stmt = text(bm25_sql.format(col_filter=col_filter)).bindparams(
                bindparam("q", value=query_text),
                bindparam("limit", value=candidate_pool),
            )
            if collection_id is not None:
                stmt = stmt.bindparams(bindparam("col_id", value=collection_id))
            res = await self.db.execute(stmt)
            bm25_rows = list(res.mappings())

        bm25_ranks: dict[int, int] = {
            row["chunk_id"]: i + 1 for i, row in enumerate(bm25_rows)
        }

        # RRF fusion
        scores: dict[int, float] = {}
        for cid, (_chunk, rank) in vec_ranks.items():
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank)
        for cid, rank in bm25_ranks.items():
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + rank)

        if not scores:
            return []

        # Build chunk index from union of sources
        chunk_index: dict[int, RetrievedChunk] = {
            cid: c for cid, (c, _) in vec_ranks.items()
        }
        for row in bm25_rows:
            cid = row["chunk_id"]
            if cid not in chunk_index:
                chunk_index[cid] = RetrievedChunk(
                    chunk_id=cid,
                    document_id=row["document_id"],
                    document_title=row["document_title"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    relevance=0.0,  # BM25-only chunks: relevance unknown in vector space
                )

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        # For chunks present in vec, preserve cosine relevance; for BM25-only, use rrf score normalized
        max_rrf = ranked[0][1] if ranked else 1.0
        result: list[RetrievedChunk] = []
        for cid, rrf in ranked:
            base = chunk_index[cid]
            if cid in vec_ranks:
                result.append(base)  # keep cosine-derived relevance
            else:
                # Synthesize a relevance proxy from RRF — useful for UI display
                synthetic = round(rrf / max_rrf, 4) if max_rrf > 0 else 0.0
                result.append(
                    RetrievedChunk(
                        chunk_id=base.chunk_id,
                        document_id=base.document_id,
                        document_title=base.document_title,
                        chunk_index=base.chunk_index,
                        text=base.text,
                        relevance=synthetic,
                    )
                )
        return result
