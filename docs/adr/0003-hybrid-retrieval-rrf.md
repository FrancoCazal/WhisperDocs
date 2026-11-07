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

## Amendment — 2026-11: the lexical leg was not BM25, and rarely matched

**What was wrong.** The lexical leg was `ts_rank_cd(tsv,
websearch_to_tsquery('simple', question))`:

- `ts_rank_cd` is Postgres cover-density ranking. It has no IDF term, so it
  is not BM25, which is what this ADR and the README called it.
- `websearch_to_tsquery` ANDs every word, and the `simple` configuration
  keeps stopwords. A chunk had to contain "how", "does" and "to" to match
  "How does HNSW compare to IVFFlat?".
- `websearch_to_tsquery` also reads "or" and "-word" in a question as
  operators.

Measured on the golden set with `scripts/retrieval_bench.py`, the lexical leg
returned rows for 5 of the 41 in-scope questions. The hybrid was vector-only
in practice.

**What changed** (alembic `0013`, `app/services/fulltext.py`):

- `chunks.tsv` is built with the configuration that matches the chunk's
  language: `wd_english`, `wd_spanish`, or `wd_simple` as the fallback. All
  three apply `unaccent` before stemming.
- The query is OR over the lexemes of the question, normalized by every
  configuration (`plainto_tsquery`, with `&` turned into `|`). No web-search
  syntax is involved.
- The leg is ranked by `ts_rank_cd` without length normalization. It is still
  not BM25. Real BM25 needs an extension (`pg_search`, `vchord_bm25`) that the
  pgvector image does not ship.
- `relevance` is now the score the list is ordered by: RRF divided by the best
  score possible, which is 1.0 when a chunk is first in every leg that returned
  rows. The vector cosine moved to its own `cosine` field, `None` for chunks
  only the lexical leg found. Before, vector hits kept their cosine and
  lexical-only hits got RRF / max RRF, so the field mixed two scales and MMR
  compared them.

**Measured** (golden set, k = 5, same query embeddings for both runs, reranker
and MMR off):

| Leg | In-scope questions with rows | Precision@5 | Recall@5 |
|---|---|---|---|
| Lexical, before | 5 / 41 | 9.3% | 8.5% |
| Lexical, after | 41 / 41 | 79.5% | 92.7% |
| Hybrid, before | 41 / 41 | 89.3% | 93.5% |
| Hybrid, after | 41 / 41 | 89.8% | 95.9% |

The ranking function was picked on this same set, out of `ts_rank` and
`ts_rank_cd` with normalization 0, 1 and 2. Each half point of precision is
one chunk out of 205. Treat the gain as "the lexical leg now contributes, and
nothing got worse", not as a tuned improvement.

**Still true.** OR matching returns rows for 8 of the 9 out-of-scope
questions too, so an empty lexical leg no longer signals "nothing relevant".
No code read it that way: in the agent the classifier routes out-of-scope
questions before retrieval, and the eval's single-turn path relies on the RAG
prompt refusing.

**MMR, measured on the same run.** With `MMR_ENABLED=true`, hybrid precision@5
falls to 26.8% at the default λ = 0.5 (recall 100%): a penalty of 0.5 outweighs
any relevance gap on a 0–1 scale, so MMR returns five different documents.
λ = 0.9 gives 62.9%. MMR stays off by default.
