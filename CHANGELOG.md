# Changelog

Notable changes, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are git tags.
Changes land through pull requests with CI green.

## [Unreleased]

Fixes from the November 2026 audit.

### Security

- The checkpointer moves to `langgraph-checkpoint` 4.x and
  `langgraph-checkpoint-postgres` 3.x. A `<3.0` cap held versions with known
  advisories, including CVE-2025-64439 (deserialization of stored
  checkpoints). Threads written by the old versions read back unchanged.
- Uploads are size-checked before the multipart body is parsed (413). A
  `.pdf` must carry the PDF header, text files must not contain NUL bytes, and
  list endpoints bound `limit`.
- Database passwords are redacted from logs, and CORS uses an allowlist
  (`CORS_ALLOWED_ORIGINS`).

### Changed

- Models: the hot-path nodes and the eval judge run on `gemini-3.8-flash`,
  since `gemini-2.5-flash` was retired on 2026-10-20. Sampling parameters are
  sent per model, and startup fails on a model ID that cannot be routed.
- Full-text retrieval ORs the question's terms with per-document language
  and accent-insensitive matching. Before, it matched 5 of 41 golden
  questions; now it matches all 41.
- `relevance` is the score results are ordered by, and the vector
  similarity moved to its own `cosine` field.
- The request's `top_k` is honored by `/agent/chat`.
- A turn with no retrieved chunks gets a fixed answer, with no model call.
  A retrieval failure gets its own answer.
- Dependencies are locked in `uv.lock`; CI and the Docker image install with
  `uv sync --frozen`.
- The frontend container serves a production build (`vite preview`), not the
  dev server. The `/consulta` single-query page is linked from the navigation.
- The README and the architecture document describe the agent as
  intent-routed and single pass: one path per turn, one retrieval, no loop.
- Answers written without the model (no documents, a failure, out of scope)
  take the language of the user's message, and the generation prompts ask for
  answers in that language. Summary headings are fixed, in English.
- The query rewrite no longer calls the model on a conversation's first
  message, where there is nothing to resolve.

### Fixed

- Compaction never leaves the conversation starting with the assistant,
  which Bedrock rejects. The summarizer no longer returns empty summaries.
- Filtered vector search no longer comes back empty for small collections.
- Document parsing runs off the event loop. An ingestion job cut short by a
  restart ends as `failed` instead of staying in progress.
- Bedrock streams retry throttling before the first token.
- The eval averages retrieval over every in-scope query, and aborts when too
  many judge calls fail.
- `GET /documents/jobs` answered 422 on every call.
- A failed answer ("I had a problem…") stays in the thread but no longer
  reaches later prompts, the summary or `query_logs`.
- A turn answered without a model call, or whose answer failed, reported the
  classifier's tokens as the answer's. The classifier's usage has its own
  metadata fields.
- A failed answer no longer lists the retrieved chunks as its citations.

### CI

- Lint, format and file hygiene (pre-commit), gitleaks, pip-audit and mypy.
- The full test suite runs against a pgvector Postgres service.
- Dependabot for Python, npm, GitHub Actions and Docker.

## [1.0] - 2026-05-19

First public snapshot, published as a single commit. After the tag, `main`
received CI, documentation and the license only.

[Unreleased]: https://github.com/FrancoCazal/WhisperDocs/compare/v1.0...HEAD
[1.0]: https://github.com/FrancoCazal/WhisperDocs/releases/tag/v1.0
