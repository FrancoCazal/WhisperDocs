# ADR 0003 — Hybrid retrieval with Reciprocal Rank Fusion

**Status:** accepted
**Date:** 2026-04

## Context

Pure vector retrieval misses critical exact-match cases:

- Acronyms, codes, and proper nouns (`HNSW`, `RRF`, `pgvector`) embed weakly.
- Short queries with named entities (`What is FastAPI?`) sometimes pull
  unrelated chunks because the embedding space puts "framework intro" content
  near every framework, not specifically FastAPI.

Pure BM25 misses paraphrasing — the original RAG failure mode.

## Decision

Hybrid retrieval: vector cosine + BM25, fused via Reciprocal Rank Fusion.

```
RRF_score(chunk) = 1 / (k + rank_vector) + 1 / (k + rank_bm25)
```

with `k = 60` (the value from the original RRF paper). Each strategy fetches
up to `candidate_pool = 20`; the top `top_k` chunks by fused score are
returned. BM25-only chunks (not in vector top-N) get a synthesized relevance
proxy from their RRF score.

Implemented in `app/services/retriever.py::VectorRetriever.hybrid_retrieve`,
exposed through `app/services/retrieval_pipeline.py::retrieve` so reranking
and MMR can layer on top.

## Why

- **RRF requires no score calibration.** Vector cosine returns ~0–1
  similarity; BM25 `ts_rank_cd` is unbounded and dataset-dependent. Combining
  raw scores requires per-corpus tuning. RRF only consumes ranks, so it
  works out-of-the-box.
- **k = 60 is robust.** The original RRF paper showed near-optimal
  performance for `k ∈ [20, 80]` across a range of IR tasks. Picking 60
  avoids hyperparameter tuning.
- **Single SQL round-trip is achievable.** Both indices live on `chunks`.
  We currently issue two queries (vector and BM25) and fuse in Python — fine
  at portfolio scale, easy to migrate to a single CTE-based query later.
- **Falls back gracefully.** Empty BM25 hits → vector-only. Empty vector
  hits (e.g., bad embedding) → BM25-only. The fusion handles either gracefully
  because RRF sums; missing rank contributes 0.

## Trade-offs

- BM25-only chunks have unknown semantic relevance. We synthesize a relevance
  proxy from RRF score for UI display; this is informative but not
  comparable to cosine relevance numerically.
- The candidate pool is fetched twice (once per strategy). Optimization
  opportunity: single CTE pulling both rankings.

## Cross-references

- Reranker (`bge-reranker-base`) operates downstream as an optional layer
  — see code `app/services/reranker.py`.
- MMR (`app/services/mmr.py`) penalizes chunks from already-selected
  documents to improve diversity at the top of the result set.
