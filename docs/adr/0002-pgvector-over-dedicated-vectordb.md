# ADR 0002 — pgvector over a dedicated vector DB

**Status:** accepted
**Date:** 2026-04

## Context

WhisperDocs needs vector similarity search over chunked documents. The market
splits into:

1. **Postgres + pgvector** — extension exposing a `vector(N)` type, IVFFlat /
   HNSW indexes, and `<=>` / `<->` / `<#>` operators.
2. **Dedicated vector DBs** — Pinecone (managed), Qdrant, Weaviate, Chroma.

## Decision

Use pgvector with HNSW indexing.

## Why

- **Single source of truth.** `documents`, `chunks`, `query_logs`, and the
  LangGraph checkpointer all live in the same Postgres. One backup story, one
  connection pool, one migration tool (Alembic). No two-system consistency
  problem when deleting a collection cascades to its chunks.
- **Hybrid retrieval works.** BM25 via `tsvector` + GIN index lives in the
  same row as the embedding. The fused query (vector + BM25 → RRF) is
  expressible in plain SQL with a CTE — no application-level join across two
  systems. See ADR 0003.
- **Filter-rich queries.** `WHERE collection_id = ?`,
  `WHERE created_at > ?`, joins to `documents.title`, etc. work natively.
  Pinecone metadata filters degrade unpredictably on low-selectivity attrs.
- **Operational scale fits.** Demo / portfolio scope: thousands to low
  millions of chunks. pgvector handles 10M+ comfortably with HNSW. Past that
  threshold, migration to Qdrant is mostly a `Reranker`-shaped abstraction
  away.
- **Cost.** Free. Pinecone's serverless free tier is generous, but adding a
  managed dependency for a portfolio project adds friction without payoff.

## Trade-offs

- Bedrock / OpenAI traffic is unaffected, but at >20M vectors the IVFFlat /
  HNSW index build becomes painful and dedicated stores start winning on raw
  recall@k.
- We initially shipped IVFFlat; it failed silently with very small datasets
  (returned 0 chunks for some collections). Migration `0007` swapped the index
  to HNSW. Lesson: HNSW is the right default in pgvector ≥0.5.

## Out of scope

Sharding strategy, replication topology — single-node Postgres is enough for
the foreseeable future.

## Amendment — 2026-11: filtered HNSW scans

"Filter-rich queries work natively" was only half true. When Postgres walks
the HNSW index in distance order, it applies `WHERE collection_id = ?` after
the scan, over only `hnsw.ef_search` candidates (40 by default). A collection
that is a small share of the corpus comes back short or empty: the same
silent failure IVFFlat had, which the swap to HNSW did not remove. With one
collection, as in the demo, it does not show. Whether it happens depends on
the plan Postgres picks, which tends to be the index walk on large tables.

Since pgvector 0.8.0, `SET LOCAL hnsw.iterative_scan = relaxed_order` keeps
walking until LIMIT rows pass the filter. `VectorRetriever.retrieve` sets it
on every query and re-sorts the rows by distance. The compose and CI images
are pinned to `pgvector/pgvector:0.8.2-pg16`, because the `pg16` tag floats.
`tests/test_retrieval_db.py` forces the index-walk plan and checks that a
3-chunk collection next to a 400-chunk one still returns its 3 chunks. It
returned none before the change.

Iterative scan is not a guarantee either. It stops at `hnsw.max_scan_tuples`
(20,000), and it can only visit elements the graph reaches. HNSW does not
promise that every element is reachable, and near-duplicate vectors, or dead
entries left by rolled-back inserts until VACUUM, make it worse. In the test
above, the 400 near-duplicate vectors left the scan out of graph after 95 to
350 rows, even with `ef_search = 400`, and the small collection came back
empty. So when a filtered search returns fewer than `top_k` rows,
`retrieve` repeats it as an exact scan: it orders by `distance + 0`, which
the planner cannot serve from the index, and Postgres reaches the
collection's chunks through `ix_documents_collection_id` and
`ix_chunks_document_id`. The cost is one distance per chunk of the
collection, paid only when the collection is small next to the corpus or has
fewer than `top_k` chunks. A second test checks the plan, so that a planner
change cannot quietly bring the index back into the exact path.
