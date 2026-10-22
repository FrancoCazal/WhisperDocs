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
