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
- Each answer says which of its calls failed or fell back, and why, for
  example "classifier · billing or quota · gemini-3.8-flash". This is `errors`
  (`{step, kind, model}`) on `/agent/chat`, the stream's final event and a
  thread's messages, and the chat shows it under the answer. When a
  provider refuses every classifier call, the turn still answers from the
  documents, so before this list nothing showed the outage.
- Each model role (classifier, query rewrite, answers, clarification, direct
  answers, compacting, eval judge, embedding) keeps the outcome of its last
  calls: when it last worked, when it last failed and why, and how many
  calls in a row failed. The record comes from real calls; nothing is called
  to check. `GET /admin/model-status` returns it, `/health?deep=true` adds it
  and turns `degraded` while a role fails, and
  `whisperdocs_model_calls_total{role,model,outcome}` counts the calls.
  While a role fails, every page shows a notice naming the roles, the
  reason and what the app does meanwhile. SETTINGS shows each role's
  status.
- `POST /retrieve`: the chunks for a query, without an answer. Each
  chunk comes with its text, its document, its relevance and its cosine,
  scoped to a collection or a single document. This is what another agent
  needs from a knowledge base, and the base of the MCP server. It runs the
  search the chat uses, and counts against the questions rate limit.
- Every operation has a fixed `operationId` (`retrieve_chunks`,
  `answer_question`, ...), where FastAPI generated
  `query_api_v1_query_post`. A test lists them all.
- An MCP server (`python -m app.mcp`, over stdio) for Claude Code, Claude
  Desktop, Cursor and other MCP clients: `search_documents` (the passages,
  nothing generated), `ask_documents` (an answer with its citations),
  `list_collections`, `list_documents`, and a resource per document with
  its full text. No tool changes the knowledge base. It runs on the
  official `mcp` SDK 2.3, speaks the 2026-07-28 spec and the older
  handshake, and each tool calls the service its REST route calls. The
  README has the configuration for each client; ADR 0005 has the design.
- History: filters by origin and by question text
  (`GET /query/history?source=&q=`), and pagination.
- Frontend tests: vitest for the API client's stream parser, and a
  Playwright test of the chat against a mocked backend.
- OpenTelemetry traces. Each turn is a trace: FastAPI's span of the request,
  the turn, a span per graph node, and under them each model call,
  embedding request and retrieval, with the GenAI semantic conventions'
  attributes (model, provider, tokens, effort, the error's kind) and the
  cost from `pricing.py`. The turn adds up its calls, the classifier's and
  the summary's included. No span holds prompt, answer or document text.
  Log lines carry `trace_id`, and `query_logs.trace_id` (migration 0016)
  links a row to its trace. Spans are exported over OTLP/HTTP only when
  `OTEL_EXPORTER_OTLP_ENDPOINT` is set; `docker compose --profile tracing`
  runs Langfuse locally to read them. ADR 0006 has the design.

### Security

- An API key (`API_KEY`, sent as `X-API-Key`). With it set, every route
  needs it except `GET /health` and the WhatsApp webhook, which checks
  Twilio's signature. `/health?deep=true` makes an embedding call and needs
  it too. Unset, the API stays open, as the local demo is. The frontend
  takes the key in SETTINGS.
- Rate limits per client and minute: questions (30), uploads (60) and eval
  runs (2), set by `RATE_LIMIT_*`. Past a limit the API answers 429 with
  `Retry-After`. An eval run takes at most 200 items.
- Error responses no longer carry exception text. Provider and database
  messages stay in the logs. `/health` reports a failed check as
  `error: <kind>`, and a failed ingestion job says what failed without
  internals.
- The checkpointer moves to `langgraph-checkpoint` 4.x and
  `langgraph-checkpoint-postgres` 3.x. A `<3.0` cap held versions with known
  advisories, including CVE-2025-64439 (deserialization of stored
  checkpoints). Threads written by the old versions read back unchanged.
- Uploads are size-checked before the multipart body is parsed (413). A
  `.pdf` must carry the PDF header, text files must not contain NUL bytes, and
  list endpoints bound `limit`.
- Database passwords are redacted from logs, and CORS uses an allowlist
  (`CORS_ALLOWED_ORIGINS`).
- WhatsApp: no log line carries a phone number. Delivery statuses and sent
  messages no longer log the recipient, a Twilio failure logs its status
  and code but not its message, and database errors leave out their
  statement's values (`hide_parameters`).
- WhatsApp: a repeated message (same `MessageSid`) is answered once, and
  each number gets the questions' limit per minute.
- The eval workflow reads the service account key from the environment
  instead of pasting the secret into its script, declares read-only
  permissions, and no longer publishes the reports, answers included, to
  `gh-pages`; they stay in the run's artifacts.

### Changed

- The conversation summary keeps where its facts came from: each user
  message records the collection or document its turn searched, and each
  fact in the summary names its source. Each pass carries over the
  previous summary's facts, within 500 tokens instead of 300.
- The eval CLI exits 3 when a run is below `AGENT_EVAL_MIN_ACCURACY` (0.80)
  or above `AGENT_EVAL_MAX_HALLUCINATION_RATE` (0.10), so the eval workflow
  fails on a regression. The workflow drops its branches for pull requests
  and schedules, which never ran, sets `AWS_REGION` (the app ignored
  `AWS_DEFAULT_REGION`) and defaults `GCP_LOCATION` to `global`.
- The WhatsApp webhook answers 204 at once and runs the turn in the
  background: Twilio waits at most 15 s, and a turn's p95 was 9.1 s. A
  number's messages are answered one at a time, in order, so a
  `/coleccion` sent before a question applies to it. One Twilio client is
  reused instead of one per message.
- Every error body is `{detail, code, request_id}`, with a stable `code`
  (`not_found`, `rate_limited`, `upstream_error`, ...). A 5xx message quotes
  the request id, which is also the `X-Request-ID` header and is on every
  log line of the request. A failed model or embedding call on `/query` is a
  502 with its `kind`, where a generation failure used to be a bare 500. An
  error the routes did not handle is a 500 the browser can read, and it no
  longer surfaces as a CORS error.
- Models: the hot-path nodes and the eval judge run on `gemini-3.8-flash`,
  since `gemini-2.5-flash` was retired on 2026-10-20. Sampling parameters are
  sent per model, and startup fails on a model ID that cannot be routed.
- Claude 4.7 and later run on Bedrock through Converse, which serves
  Claude Sonnet 5.5 (`global.anthropic.claude-sonnet-5-5`), the replacement
  for Sonnet 4.5 (Legacy on Bedrock, end of life on 2027-04-08). They get
  the effort in `BEDROCK_CLAUDE_EFFORT` (default `low`), room in
  `max_tokens` for their thinking, and structured output as a JSON schema,
  since Sonnet 5.5 rejects a forced tool call. The defaults stay on Sonnet
  4.5 until the eval has run on 5.5.
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

- A failed summary is no longer retried on every turn: compaction waits
  `AGENT_COMPACTION_RETRY_MESSAGES` (6) messages first. A summary cut at
  `max_tokens`, or empty, is not kept; before, a cut one replaced the
  previous summary.
- The checkpoint tables no longer grow with the square of a thread's turns.
  Each turn saved about five checkpoints, and each change to `messages`
  stored the whole list again: 120 turns kept 70 MB of copies of 0.6 MB of
  messages. A background sweep keeps each thread's latest checkpoint, every
  `AGENT_CHECKPOINT_PRUNE_INTERVAL_S` (600 s).
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
