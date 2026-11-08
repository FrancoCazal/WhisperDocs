# Changelog

Notable changes, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are git tags.
Changes land through pull requests with CI green.

## [Unreleased]

Fixes from the November 2026 audit, and a review of the frontend.

### Added

- Chat: each `[n]` citation shows its passage on hover and opens the
  sources, with a link to the chunk in its document. The thread is in the
  URL. A streaming answer shows its progress and can be stopped; a failed
  one is an error card with Retry. Each turn shows its cost, tokens and
  time, and the header the thread's total.
- Threads have titles (`conversations.title`, migration 0014), a search
  and a saved filter, and notes can be listed by thread.
- Evaluation: a run's answers and verdicts item by item
  (`GET /eval/results/{run_id}`), a diff between two runs, and a cost
  confirmation before a run starts.
- Documents: the document page can delete or move the document, search its
  chunks and ask the chat about it: a thread that searches only that
  document (`conversations.document_id`, migration 0015; `document_id` on
  `/agent/chat`). Uploads are checked before they are
  sent and show their progress.
- History: filters by origin and by question text
  (`GET /query/history?source=&q=`), and pagination.
- Frontend tests: vitest for the API client's stream parser, and a
  Playwright test of the chat against a mocked backend.

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
- The frontend is a static single-page app, served by nginx in its container
  (port 8080), with long-lived caching for the hashed assets. The pages'
  paths are in English (`/query`, `/documents`, `/collections`, `/history`,
  `/evaluation`); the old Spanish paths redirect.
- The frontend ships its own font, and loads the code highlighter with the
  first code block. Answers are at reading size.
- The compose database's port on the host is `DB_PORT`, 5435 by default:
  another Postgres on 5435 kept it from starting.
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
- The documents page's bulk delete could delete documents outside the
  filter. A streamed answer could land in another thread. A document's page
  could not be opened.
- The UI's costs came from an old price table; they are the backend's
  `cost_usd`. Lists stopped at 50 rows. Model output could load remote
  images.
- After a Postgres restart the API kept handing out dead connections.
- A question whose stream was cut off was left without an answer, and broke
  every later prompt of its thread.
- `GET /eval/results` read the reports on the event loop, and applied
  `limit` before it skipped the files that are not reports.
- The eval CLI's reports showed metrics an old run did not save as 0.
- On a phone, iOS zoomed into a search box, a select or TOP_K when it took
  focus: every field is 16px there now.
- When the classifier's model failed, every question got "Can you give me
  more details?": the clarification it fell back to calls a model too, and
  failed the same way. A classifier failure now searches the documents.

### CI

- Lint, format and file hygiene (pre-commit), gitleaks, pip-audit and mypy.
- The full test suite runs against a pgvector Postgres service.
- Dependabot for Python, npm, GitHub Actions and Docker.
- A frontend job: types, lint and format, build, and the unit and browser
  tests.
- No test can reach a model or embedding API, even with credentials in
  `.env`: a test that resolves a host other than this machine or the test
  database fails.

## [1.0] - 2026-05-19

First public snapshot, published as a single commit. After the tag, `main`
received CI, documentation and the license only.

[Unreleased]: https://github.com/FrancoCazal/WhisperDocs/compare/v1.0...HEAD
[1.0]: https://github.com/FrancoCazal/WhisperDocs/releases/tag/v1.0
