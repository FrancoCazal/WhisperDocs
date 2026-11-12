# WhisperDocs

[![Tests](https://github.com/FrancoCazal/WhisperDocs/actions/workflows/tests.yml/badge.svg)](https://github.com/FrancoCazal/WhisperDocs/actions/workflows/tests.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A self-hosted RAG engine with an intent-routed agent: it answers questions about your own documents with citations, and measures how good and how expensive those answers are. For teams that need answers from their own documentation, in a web chat or over WhatsApp, with numbers on quality and cost. What sets it apart: an eval suite with diffs between runs, the USD cost of each RAG answer, and a model chosen per graph node.

> **Portfolio project — runs locally via Docker. No hosted deployment.** Clone, fill `.env`, `docker compose up`.

> **Published as a snapshot.** WhisperDocs was built between April and May 2026 and pushed here in a single commit, so this repository does not carry the development history. The design decisions are recorded in [`docs/adr/`](docs/adr/).

![System overview dashboard](media/whisperdocs1.png)

**Stack** · FastAPI · PostgreSQL + pgvector · LangGraph · Gemini + AWS Bedrock (Claude) · OpenAI Embeddings · Twilio · React + TanStack Router · structlog · Prometheus · OpenTelemetry + Langfuse

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for component diagrams, sequence flows, and data model. Decision records live under [`docs/adr/`](docs/adr/).

---

## Highlights

- **Intent-routed RAG agent, single pass (LangGraph)** — a classifier sends each turn down one path: search, clarify, direct answer or out-of-scope. Search rewrites the question, retrieves once and generates; there is no loop or second retrieval, by design. 7 nodes in an explicit state graph (compaction and summary keep long threads bounded), each testable in isolation; `AsyncPostgresSaver` handles persistence.
- **Hybrid retrieval** — vector cosine via pgvector + Postgres full-text search (OR over the question's stemmed terms, per-document language, accent-insensitive, ranked by `ts_rank_cd`), fused with Reciprocal Rank Fusion. Optional cross-encoder rerank (`bge-reranker-base`) + document-level MMR for diversity.
- **Multi-provider routing** — Gemini 3.8 Flash for cheap hot-path nodes (classifier, query rewrite, direct answers); AWS Bedrock Claude for grounded RAG generation. Provider auto-detected from model ID; startup fails on a model ID that can't be routed.
- **Token-level SSE streaming** — `astream_events` v2 + custom event dispatch. Frontend renders tokens as they arrive, with progress markers per node.
- **Async ingestion with progress** — `IngestionJob` row + asyncio worker. Per-stage progress (extracting → chunking → embedding → persisting). Frontend bar updates over SSE. Parsing runs off the event loop, and a job cut short by a restart ends as `failed`, not stuck.
- **Seven formats, citations that say where** — PDF, DOCX, XLSX, CSV, HTML, Markdown and plain text, all extracted to one shape (headings, markdown tables, page breaks). Chunks follow the document's sections and keep tables whole or repeat their header; each citation names its section and, for a PDF, its pages.
- **Eval suite (50-query golden set)** — single-turn retrieval + generation path. LLM-as-judge for correctness + faithfulness (unsupported claims), retrieval precision/recall@k, latency p50/p95, generation cost USD. Diff between runs for regression detection.
- **Cost tracking** — per-call USD via `pricing.py`, persisted on `query_logs`. Single source for runner, API, dashboard.
- **Logs, metrics and traces** — `structlog` JSON output, Prometheus `/metrics` (HTTP rate, errors and latency per route, turn duration, tokens, retrieval relevance, errors) with a Grafana dashboard and alert rules, and an OpenTelemetry trace per turn: each graph node, model call, embedding and retrieval, with its tokens and cost, under the GenAI semantic conventions. A local Langfuse shows them (see [Tracing](#tracing)).
- **WhatsApp integration** — full agent pipeline over Twilio webhooks, answered in the background so Twilio's 15 s timeout never cuts a turn.
- **MCP server** — the knowledge base for Claude Code, Claude Desktop and Cursor:
  search, answers with citations, fields extracted from a document with the
  words that state each, the lists and a resource per document, over stdio or
  over HTTP at `/mcp`, where each key reaches only its collections. A key made
  to write also adds, deletes and moves documents, after the user confirms a
  deletion or a move, and every write is in an audit log. Each tool calls the
  same service as its REST route.
- **MCP connector** — WhisperDocs as an MCP client, for ingestion: it reads an
  MCP server's resources into a collection and keeps them in step on later
  runs, re-embedding only what changed.

---

## Screens

### Multi-turn chat with grounded citations

Inline `[n]` citations resolve to the source documents pulled by the retrieval pipeline.
Per-turn metadata footer shows latency, input/output tokens, model, and USD cost.

![Chat answer with citations](media/whisperdocs2.png)

### Retrieval transparency

Every turn exposes the chunks that were actually retrieved — text, source document, and
relevance: the fused RRF score scaled to 0–1, or the cross-encoder score when reranking is on.
Lets you debug retrieval without leaving the chat.

![Retrieved chunks side panel](media/whisperdocs4.png)

### Indexed corpus

Drop PDF / MD / TXT files into any collection. Each file streams through the async
ingestion pipeline; chunk and token counts are visible per document.

![Documents library](media/whisperdocs3.png)

---

## Quickstart

```bash
git clone <this-repo> && cd WhisperDocs
cp .env.example .env
# See "Minimum to run" below for which keys to fill.
docker compose up --build
```

Postgres (with pgvector) runs in the compose stack; no host install needed. `DATABASE_URL` already points at the in-network DB and Alembic migrations run automatically on container start.

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| Frontend | http://localhost:8080 |
| `/metrics` | http://localhost:8000/api/v1/metrics |

### Minimum to run

| Required | Why |
|---|---|
| `OPENAI_API_KEY` | Embeddings (`text-embedding-3-small`) — no fallback. |
| **One generation provider:** `GCP_PROJECT` *(Vertex AI / Gemini)* **or** `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` *(Bedrock Claude)* | Agent nodes route by model ID. Default config uses Gemini for hot-path nodes and Bedrock for RAG generation; override via `AGENT_MODEL_*` env vars to consolidate on one provider. |

Twilio, reranker, and MMR are all optional. Without Twilio the WhatsApp channel just stays inert; the chat UI works.

> **Gemini auth tip**: Vertex AI uses Application Default Credentials. Run `gcloud auth application-default login` on the host — `docker-compose.yml` mounts the resulting ADC file into the API container.

### Local dev

```bash
uv sync --frozen --extra dev        # .venv with the exact versions in uv.lock
cp .env.example .env
uv run alembic upgrade head
uv run uvicorn app.main:app --reload

cd frontend && npm install && npm run dev
```

Dependencies are locked in `uv.lock`; CI and the Docker image install it with
`uv sync --frozen`. After changing `pyproject.toml`, run `uv lock` and commit both files.

The `Dockerfile` has two images, both running as a non-root user (`app`, uid 1000) with a
`HEALTHCHECK` on `/api/v1/health`:

- **`runtime`**, the default (`docker build .`): the dependencies and the code, with no
  compilers and no dev tools. It is about 0.9 GB. The reranker's extra:
  `--build-arg EXTRAS="--extra rerank"`.
- **`dev`**, which compose builds: `runtime` plus pytest, ruff and mypy, so the tests run in the
  container.

Both carry `fixtures/`, so `python -m scripts.seed_eval` works without compose's bind mounts.

---

## Environment

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5442/whisperdocs
DB_PORT=5442                     # the compose database's port on the host; change it if 5442 is taken
OPENAI_API_KEY=sk-...

# Generation providers
GCP_PROJECT=...                  # Gemini via Vertex AI; auth via `gcloud auth application-default login`
GCP_LOCATION=global              # gemini-3.8-flash is served from global, us or eu only
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Optional
API_KEY=...                      # required in X-API-Key; unset = open API (see Security posture)
RATE_LIMIT_PER_MINUTE=30         # questions per client a minute; 0 = off
TWILIO_ACCOUNT_SID=...           # WhatsApp
TWILIO_AUTH_TOKEN=...
RERANKER_ENABLED=false           # cross-encoder reranking (needs .[rerank])
MMR_ENABLED=false                # source diversity
AGENT_STREAMING_TOKENS=true      # token-level SSE
OTEL_EXPORTER_OTLP_ENDPOINT=...  # OTLP/HTTP base URL for traces; unset = they stay in the process
OTEL_EXPORTER_OTLP_HEADERS=...   # e.g. the viewer's auth (see Tracing)
```

See `.env.example` for every knob.

---

## API surface

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/collections` | Create a collection |
| `GET` | `/api/v1/collections` | List collections |
| `POST` | `/api/v1/documents/upload` | Sync ingest (small docs) |
| `POST` | `/api/v1/documents/upload-batch` | Sync batch ingest |
| `POST` | `/api/v1/documents/upload-async` | Async ingest, returns `job_id` |
| `GET` | `/api/v1/documents/jobs` | List ingestion jobs |
| `GET` | `/api/v1/documents/jobs/{job_id}/stream` | SSE: ingestion progress |
| `POST` | `/api/v1/documents/bulk-delete` | Delete many documents |
| `POST` | `/api/v1/documents/bulk-move` | Move documents between collections |
| `POST` | `/api/v1/documents/{document_id}/extract` | Fields a JSON Schema names, each with the chunk and the words that state it |
| `POST` | `/api/v1/query` | Single-turn RAG query |
| `POST` | `/api/v1/retrieve` | The chunks for a query, without an answer (for other agents) |
| `GET` | `/api/v1/query/history` | Paginated query log |
| `POST` | `/api/v1/agent/chat` | Multi-turn agent chat |
| `POST` | `/api/v1/agent/chat/stream` | SSE: node + token events |
| `GET` | `/api/v1/conversations` | List conversations |
| `GET` | `/api/v1/conversations/{thread_id}/messages` | Conversation history |
| `PATCH` | `/api/v1/conversations/{thread_id}` | Update conversation (e.g. `is_saved`) |
| `DELETE` | `/api/v1/conversations/{thread_id}` | Delete conversation |
| `GET` | `/api/v1/notes` | List saved notes |
| `DELETE` | `/api/v1/notes/{note_id}` | Delete saved note |

| `POST` | `/api/v1/webhook/whatsapp` | Twilio inbound webhook |
| `POST` | `/api/v1/webhook/whatsapp/status` | Twilio delivery-status callback |
| `POST` | `/api/v1/eval/run` | Eval suite |
| `GET` | `/api/v1/eval/results` | List past eval reports |
| `GET` | `/api/v1/admin/settings` | Runtime settings snapshot |
| `GET` | `/api/v1/admin/model-status` | Each model role's last calls, from real traffic (no model is called) |
| `GET` | `/api/v1/admin/audit-log` | The MCP server's writes, newest first, done or not |
| `GET` | `/api/v1/health` | Health check (`?deep=true` for full probe and each model role's state); 503 when degraded |
| `GET` | `/api/v1/metrics` | Prometheus metrics |

Uploads accept PDF, DOCX, XLSX, CSV, HTML, Markdown and plain text, up to 10 MB
per file. Batches take up to 20 files and 50 MB in total. A larger body is
rejected with 413 before it is parsed. A `.pdf` must carry the PDF header, a
`.docx` or `.xlsx` must be a zip archive, a text format must not contain
binary data, and file names and titles are capped at 300 characters. List
endpoints take `limit` between 1 and 200.

What each format becomes:

| Format | Extracted as | A citation names |
|---|---|---|
| PDF | each page's text, each table once as a markdown table where it sits | its pages |
| DOCX | paragraphs and tables in order; Title and Heading 1-6 as headings; list items | its section |
| XLSX | each sheet as a heading over a markdown table of its cells | its sheet |
| CSV | one markdown table, the delimiter sniffed | — |
| HTML | headings, paragraphs, lists, tables and `<pre>`; no scripts, styles or markup | its section |
| Markdown | as written | its section |
| Plain text | as written | — |

Text files are read as UTF-8, by their byte order mark (UTF-16, UTF-32), or as
the encoding `charset-normalizer` finds. Not read: scanned PDFs (there is no
OCR, and one with no text layer is a 422), legacy `.doc` and `.xls`, DOCX
headers, footers, footnotes and text boxes, and XLSX formulas never
calculated. A DOCX or XLSX that would expand past 100 MB is refused unopened,
and so is a document past 20 million characters of text.

Chunks hold up to 500 tokens. A heading starts a new one once the current one
has 250 tokens, so short sections share a chunk; each chunk starts with the
headings above it, a table that does not fit is cut between rows with its
header repeated, and a paragraph longer than a chunk is cut at a sentence or
word with 50 tokens of overlap. Chunks stored before this have no section or
pages: upload the document again to get them.

A collection holds each document once: uploading a file whose extracted text
the collection already holds is a 409 that names the document holding it,
decided before anything is embedded, and a move into a collection that holds
the same text leaves that document where it is, with the reason in `errors`.
Every document keeps all its chunks; a chunk whose text another document has
too reuses that chunk's vector when the same embedding model made it, and a
text a document repeats is embedded once.

Full reference at `/docs` (FastAPI auto-generated OpenAPI). Every operation has a
fixed `operationId` (`retrieve_chunks`, `answer_question`, `list_collections`, ...),
so a client generated from the schema, or an agent's tool list, keeps its names;
`tests/test_openapi.py` lists them all.

---

## MCP server

WhisperDocs is also an MCP server, so Claude Code, Claude Desktop, Cursor or
any other MCP client can use the knowledge base. It runs over stdio, against
the compose Postgres, or over Streamable HTTP at `/mcp`, inside the API,
where each key reaches only its collections.

| Tool | What it returns |
|------|-----------------|
| `search_documents` | The passages for a query, with their text and document, and nothing generated: the client answers with its own model. Scoped by `collection_id` or `document_id`; `top_k` from 1 to 20. |
| `ask_documents` | WhisperDocs' own answer, with its citations: `[n]` in the answer is `citations[n-1]`. It pays for a model call and is saved to the query history. |
| `extract_fields` | The fields a JSON Schema names, out of one document, each with the chunk and the words that state it; null when the document does not say. It pays for a model call. |
| `list_collections` | The collections, with their document, chunk and token totals. |
| `list_documents` | A page of documents, each with its resource URI. |
| `ingest_document` | Adds a file to a collection, as an upload does: text for `.md`, `.txt`, `.csv` and `.html`, base64 for `.pdf`, `.docx` and `.xlsx`, up to 10 MB. |
| `delete_document` | Deletes a document, once the user confirms it. |
| `move_document` | Moves a document to another collection, once the user confirms it. |

Each document is also a resource, `whisperdocs://documents/{id}`, holding its
full text. Each tool calls the service its REST route calls (`/retrieve`,
`/query`, `/documents/{id}/extract`, the upload, the lists, the delete and the
move), so both return the same thing. The server speaks the current MCP spec
(2026-07-28) and the older `initialize` handshake that many clients still use.
[ADR 0005](docs/adr/0005-mcp-server.md) has the design, and
[ADR 0008](docs/adr/0008-mcp-writes-confirmed-and-audited.md) the writes.

With the stack up (`docker compose up`; after pulling this change, rebuild
the API image once with `docker compose build api`), point the client at the
API container. Use the absolute path to this repository's
`docker-compose.yml`.

**Claude Code**

```bash
claude mcp add whisperdocs -- docker compose -f /path/to/WhisperDocs/docker-compose.yml exec -T api python -m app.mcp
```

**Claude Desktop** (`claude_desktop_config.json`) and **Cursor**
(`~/.cursor/mcp.json`, or `.cursor/mcp.json` in a project) take the same
entry:

```json
{
  "mcpServers": {
    "whisperdocs": {
      "command": "docker",
      "args": [
        "compose", "-f", "/path/to/WhisperDocs/docker-compose.yml",
        "exec", "-T", "api", "python", "-m", "app.mcp"
      ]
    }
  }
}
```

Without Docker, `uv run python -m app.mcp` starts it from a checkout whose
`DATABASE_URL` reaches the database (`localhost:5442` with the compose
defaults).

- **No auth over stdio.** The client starts the process on your machine, and
  the process has the database and keys the API has, and every collection.
- **Its own process.** The API's rate limits and its model status do not see
  the stdio server's calls; the server logs to stderr, which the client keeps
  as its MCP log.

### Over HTTP, with a key per client

The API serves the same server at `http://localhost:8000/mcp` (Streamable
HTTP, stateless). Give each client its own key, for the collections it
should see ([ADR 0007](docs/adr/0007-mcp-over-http-scoped-keys.md)):

```bash
docker compose exec api python -m app.mcp.keys create claude-code --collection 3
# wdk_...   shown once: only its SHA-256 is stored
docker compose exec api python -m app.mcp.keys create cursor --collection 3 --collection 5
docker compose exec api python -m app.mcp.keys create editor --collection 3 --write
docker compose exec api python -m app.mcp.keys create admin --all
docker compose exec api python -m app.mcp.keys list
docker compose exec api python -m app.mcp.keys revoke cursor
```

**Claude Code**

```bash
claude mcp add --transport http whisperdocs http://localhost:8000/mcp \
  --header "Authorization: Bearer wdk_..."
```

**Cursor** (`mcp.json`):

```json
{
  "mcpServers": {
    "whisperdocs": {
      "url": "http://localhost:8000/mcp",
      "headers": { "Authorization": "Bearer wdk_..." }
    }
  }
}
```

- **A key reaches its collections.** A collection or document out of its
  reach gets the error a missing one gets. The lists, `resources/list` and
  the document resource show only what it reaches. A search without
  `collection_id` takes the key's collection when it has one. `--all`
  reaches every collection, today's and later ones.
- **The API's key.** `API_KEY` also works at `/mcp`, and reaches every
  collection. Unset, `/mcp` is open, as the API is: a client with no key
  reaches everything. Set, a request without a valid key is a 401. A key
  goes in `Authorization: Bearer` or `X-API-Key`.
- **Shared limits.** `search_documents`, `ask_documents` and `extract_fields`
  count against the API's questions limit, per key; `ingest_document`
  against the uploads limit.
- **localhost only.** `/mcp` answers the Host headers in
  `MCP_ALLOWED_HOSTS` (localhost by default), the SDK's guard against DNS
  rebinding. Add the API's name to reach it by that name.

### Writes: a key that writes, the user's yes, and a log

- **A key reads only** unless made with `--write`; then it adds, deletes and
  moves documents in the collections it reaches, and a move needs both. The
  API's own key, an open API and stdio write, as the REST API lets them.
- **The user confirms** a deletion or a move, through MCP elicitation: the
  client shows the question ("Delete "a.md" (document 7, 12 chunks) from the
  collection "Docs"? It cannot be undone.") and the server acts only on a
  yes. Claude Code, Cursor and VS Code declare elicitation; Claude Code
  serves it in interactive sessions. A client that cannot ask its user is
  refused, and so is a client on MCP 2025-11-25 over HTTP, whose stateless
  exchange has no channel for the question; over stdio either era works.
  `ingest_document` asks nothing: it only adds, and refuses a text the
  collection holds.
- **Every write is logged**, done or not: who (the key's name, `api-key`,
  `open` or `stdio`), the tool, the document and collection, and whether it
  was done, declined, cancelled, refused or failed, with why.
  `GET /api/v1/admin/audit-log` pages the log, newest first.

### Reading another MCP server's resources

`python -m app.connectors.mcp_sync` turns WhisperDocs into an MCP client that
reads a server's resources into a collection. A later run leaves the ones
whose text did not change alone, replaces the ones that did, re-embedding only
the chunks that changed, and adds the new ones; it deletes nothing. No model
is involved.

```bash
# A server over stdio: its command goes after --. --env passes it a variable.
python -m app.connectors.mcp_sync --collection 3 --dry-run -- npx -y @scope/some-server
# Over HTTP, with a token from an environment variable.
python -m app.connectors.mcp_sync --collection 3 --url https://host/mcp --bearer-env SERVER_TOKEN
# Another WhisperDocs, its documents only.
python -m app.connectors.mcp_sync --collection 3 --url http://other:8000/mcp \
  --bearer-env OTHER_KEY --prefix whisperdocs://documents/
```

It reads what `resources/list` names (`--prefix` narrows it), or only the
URIs given with `--uri` or `--uri-file` (one a line). The reference
filesystem server lists nothing on purpose, and reads any file under its
directories by URI, so pass it the files:
`--uri-file uris.txt -- npx -y @modelcontextprotocol/server-filesystem /docs`,
with lines such as `file:///docs/runbook.md`. `--dry-run` compares and embeds
and writes nothing, and needs no embedding key. Each line of output is a
resource: added, updated, unchanged, skipped (a format WhisperDocs does not
read, a text the collection already holds, too large) or failed.

---

## Tracing

Each turn is an OpenTelemetry trace: the HTTP request, the turn, a span per
graph node, and under them each model call, embedding request and retrieval.
A model call records its model, provider, token counts, cost (from
`pricing.py`), effort or temperature when sent, and the kind of error when it
fails; the turn adds up its calls. Spans carry no prompt, answer or document
text. The attribute names follow the OpenTelemetry GenAI semantic conventions
(see [ADR 0006](docs/adr/0006-opentelemetry-traces.md)).

Every log line inside a turn carries its `trace_id`, and so does the turn's
row in `query_logs`.

To read the traces in a local [Langfuse](https://langfuse.com):

```bash
docker compose --profile tracing up -d    # Langfuse on http://localhost:3000
```

Add these two lines to `.env` and restart the API
(`docker compose up -d api`):

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://langfuse-web:3000/api/public/otel
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20cGstbGYtd2hpc3BlcmRvY3MtbG9jYWw6c2stbGYtd2hpc3BlcmRvY3MtbG9jYWw=,x-langfuse-ingestion-version=4
```

Sign in as `admin@whisperdocs.local`, password `whisperdocs-local`. The
compose file creates the project and its keys on the first start; the header
is their Basic auth. These are local-only credentials. Any other OTLP/HTTP
backend works the same way: Phoenix, Jaeger, an OpenTelemetry Collector or
Langfuse Cloud. Without an endpoint, no span leaves the process.

The History page shows each logged turn's `trace_id` and prompt version. To
open the trace from there, give the frontend the trace page's URL, with
`{trace_id}` for the id, and rebuild it (`docker compose build frontend`; for
`npm run dev`, set it in `frontend/.env`):

```bash
VITE_TRACE_URL=http://localhost:3000/project/whisperdocs/traces/{trace_id}
```

### Dashboard and alerts

`ops/` holds a Grafana dashboard and Prometheus alert rules over `/metrics`:
HTTP rate, 5xx share and p95 by route; turns per action and their latency;
model calls, failures and tokens; retrieval relevance.

```bash
docker compose --profile metrics up -d    # Grafana on http://localhost:3001, Prometheus on :9090
```

The dashboard opens without signing in (`admin` / `whisperdocs-local` to edit).
The alerts (the API unreachable, over 5% of 5xx, a model role failing, slow
turns) show at http://localhost:9090/alerts; route them with an Alertmanager
in a deployment. With `API_KEY` set, Prometheus needs the key: see
[`ops/prometheus/prometheus.yml`](ops/prometheus/prometheus.yml).

---

## Security posture

Out of the box, WhisperDocs runs as a local demo: the API has no key, and
CORS admits only the local frontends. Before the API can be reached from any
other machine, set `API_KEY`.

- **API key.** With `API_KEY` set, every route needs it in the `X-API-Key`
  header, with two exceptions. `GET /health` is a probe and needs no key. The
  WhatsApp webhook checks Twilio's signature instead. `/health?deep=true`
  makes an embedding call, so it does need the key. The frontend asks for the
  key in SETTINGS and keeps it in the browser's local storage. It is one
  shared key: it controls access, but it is not per-user authentication.
- **MCP keys.** `/mcp` also takes keys of its own, each scoped to some
  collections and revocable alone (`python -m app.mcp.keys`; see
  [Over HTTP](#over-http-with-a-key-per-client)). Only their SHA-256 is
  stored.
- **Rate limits.** Each client gets a limit per minute: 30 questions
  (`/query`, `/retrieve`, `/agent/chat*`, `/health?deep=true`), 60 uploads and 2 eval
  runs. The client is the key when auth is on, the IP otherwise. Past a
  limit, the API answers 429 with `Retry-After`. The counts live in the API
  process's memory: one uvicorn process, starting over on a restart. Behind a
  reverse proxy every client has the proxy's IP, unless uvicorn trusts the
  proxy's `X-Forwarded-For` (`--forwarded-allow-ips`). An eval run takes at
  most 200 items.
- **Errors.** Every error body is `{detail, code, request_id}`. A 5xx message
  quotes its request id, and that id finds the call in the logs. It is also
  the `X-Request-ID` header, which a caller can send. Provider and database
  messages stay in the logs. A failed model or embedding call is a 502 with
  its `kind`: `billing`, `auth`, `rate_limit`, `timeout`, `unavailable`,
  `invalid_request`, `bad_response` or `unknown`.
- **Uploads.** Size limits are checked before the body is parsed (see above).
- **WhatsApp.** The webhook checks Twilio's signature, drops a message it
  already took (same `MessageSid`) and a number past `RATE_LIMIT_PER_MINUTE`
  messages a minute, and answers 204 at once. The turn runs in the
  background, one message at a time per number, in the order they came. No
  log line carries a phone number: threads are a UUID5 of it, Twilio's error
  messages are left out, and database errors hide their statement's values.
- **Prompt injection.** Documents, the summary of earlier turns and the
  transcript the summarizer reads reach the models as data, each inside an
  element it cannot close. Ingestion flags each chunk whose text reads as an
  instruction to the model, in English or Spanish: `instruction_flags` on
  `GET /documents/{id}/chunks`, `flagged_chunks` on the document, and an
  `instruction_like_text` warning in the logs. A flag is a lead, not a
  verdict: a guide to prompting quotes injections. The Documents page marks
  a flagged document, and its page names each chunk's rules. `python -m
  scripts.flag_instructions` sets the flags again on stored chunks, after a
  rule changes or for documents from before the flags. The adversarial set
  below measures what gets through.
- **Personal data in the query log.** `query_logs` keeps each question and
  answer with emails, phone numbers, card numbers (Luhn), CBUs, CUITs/CUILs
  and DNIs replaced by a placeholder (`[email]`, `[phone]`, ...). The user
  still gets the answer as written, and the thread keeps the words as typed,
  since the agent reads them back.
- **Not covered.** There are no user accounts or roles: whoever has the key
  can read and delete every document and thread. The API serves no TLS, so
  put a TLS proxy in front of it. Documents and questions go to the
  configured providers: OpenAI for embeddings, Vertex AI or Bedrock for
  generation.

---

## Eval suite

Reproducible evaluation against a 50-query golden set covering all seed fixtures.

```bash
# Seed the eval corpus. A file the collection holds already is skipped, so a collection
# seeded before keeps its chunks: after a change to extraction or chunking, delete it
# first (the Collections page), then seed it again.
python -m scripts.seed_eval

# Run the suite — resolves collection by name; writes JSON + Markdown.
python -m app.eval.runner run --markdown

# Compare runs.
python -m app.eval.runner diff \
  eval/results/<prev>.json eval/results/<curr>.json -o eval/results/diff.md
```

What it measures: the single-turn retrieval + generation path
(`retrieval_pipeline.retrieve` and the single-turn RAG prompt in `app/services/rag.py`), called
directly. It does not go through the agent graph, so there is no classifier, query rewrite or
conversation history; routing is covered by the graph and router unit tests in `tests/agent/`.

Metrics: accuracy, unsupported-claim rate, retrieval precision@k and recall@k (averaged over the
in-scope queries), latency avg/p50/p95, and generation cost in USD (RAG model only; judge calls
and embeddings are not included). Bucketed by difficulty and category.

An unsupported claim is one the retrieved context does not support, true or not: the rate
measures faithfulness to the context, not facts. A report gives it over the in-scope items,
since the out-of-scope ones are mostly refusals with nothing to support and dilute it, and over
all items. The all-items figure is the one the API (`hallucination_rate`), the thresholds
(`AGENT_EVAL_MAX_HALLUCINATION_RATE`, `--max-hallucination`) and `runner diff` read, and it keeps
that name there so stored reports and scripts still work.

The run aborts with exit
code 2 when more than 10% of the judge calls fail (`AGENT_EVAL_MAX_JUDGE_ERROR_RATE`), instead of
scoring those items as failures. A finished run exits 3 when its accuracy is below 80% or its
unsupported-claim rate (all items) above 10% (`AGENT_EVAL_MIN_ACCURACY`,
`AGENT_EVAL_MAX_HALLUCINATION_RATE`, or
`--min-accuracy` and `--max-hallucination`), so a script or CI job fails on a regression. The
five full runs of 2026-04-28 scored 88-96% and 2-6%.

The `Eval suite` workflow runs it from GitHub Actions, by hand only, since it calls paid APIs:
`smoke` (10 items) or `full`. It needs the provider secrets (`OPENAI_API_KEY`, `GCP_PROJECT`,
`GCP_SA_KEY_JSON`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`); the reports go to the run's
artifacts.

### How the judge rules

Two judge calls score an answer, and each one reasons before it rules.

- **Correctness.** An in-scope item lists the `required_facts` a correct answer must state and
  the `optional_facts` it may add. The judge goes through the required facts one by one, says
  which the answer states (in any wording) and which of its statements contradict one, and the
  verdict follows in code: it passes only when every required fact is stated and nothing
  contradicts one. A report names what is missing (`facts_missing`) and what counts against the
  answer (`wrong_claims`), so a failure says why. An answer is not failed for lacking a detail
  nobody listed, which is what the earlier whole-answer comparison did to "What are the two
  phases of a RAG system?" for not detailing the second one. An item with no facts is still
  compared with its `expected_answer` as a whole, and an out-of-scope item is judged on whether
  the system refused.
- **Grounding.** The judge lists the claims in the answer that the context does not support,
  whether or not they are true; the answer is grounded when it lists none. `[n]` citation
  markers are not claims.

Whether this judge rules as a careful reader would is measured, not assumed. `label` draws a
sample of a run's verdicts, balanced by what the judge ruled so the failures are in it, and
writes what the judge read without what it decided; you fill in `human` (`true` when the answer
is correct, or fully grounded) on each line; `agreement` sets your rulings against the judge's
and reports the agreement, Cohen's kappa and the verdicts you split on. No model is called,
and `label` refuses to overwrite a file of rulings.

```bash
python -m app.eval.runner label eval/results/<run>.json --n 50 -o eval/results/labels.jsonl
# fill "human" in each line of eval/results/labels.jsonl
python -m app.eval.runner agreement eval/results/<run>.json eval/results/labels.jsonl
```

### Adversarial set

Eight questions over four documents that carry prompt injections (an
instruction in an HTML comment, a forged end of the documents with a new
question, a fake system message, a phishing link), and two questions that are
injections themselves. They run apart from the golden set, over their own
collection:

```bash
python -m scripts.seed_eval --root fixtures/adversarial-docs --collection-name "Eval Adversarial"
python -m app.eval.runner run --dataset eval/golden/adversarial.jsonl \
  --collection-name "Eval Adversarial" --markdown
```

Each item lists in `forbidden` what its injection asks for ("ACCESS GRANTED",
a password, the system prompt's words, the link); an answer containing any of
it fails without a judge call. The `Eval suite` workflow runs the set with its
`adversarial` mode.

### Prompt versions and the prompt gate

`PROMPT_VERSION` (`app/agent/prompt_version.py`) is 12 hex characters of a hash over every
constant in `app/agent/prompts.py` and the schemas of the two structured outputs. It changes
when a prompt does, and only then. Each `query_logs` row (`prompt_version`), each turn's trace
(`whisperdocs.prompt_version`) and each eval report carry it; a report also carries the judge's
version, from `app/eval/judges.py`.

`runner diff` compares two reports over the same items and exits 3 when the current one's
accuracy is more than 10 points below the previous one's, or its hallucination rate more than
10 points above (`AGENT_EVAL_MAX_ACCURACY_DROP`, `AGENT_EVAL_MAX_HALLUCINATION_RISE`, or
`--max-accuracy-drop` and `--max-hallucination-rise`). Five 50-item runs of one configuration
spread 8 and 4 points. The diff names both prompt versions and warns when the judge's changed;
so does COMPARE on the Evaluation page.

The `Prompt gate` workflow does this on a pull request that changes `prompts.py`, the agent's
schemas or `app/services/rag.py`: it answers the golden set with the base's code and with the
head's, then runs the diff. Each run calls the paid APIs twice, so it is off until the
repository variable `PROMPT_GATE` is `on`; `PROMPT_GATE_ITEMS` answers fewer items for less.

### Comparing runs

One run is one draw: the same code and settings land a few points apart from run to run, and
the judge can rule the other way on the same answer. So:

- **Run config.** Each report records what it ran with: the commit (and whether the tree had
  uncommitted changes), a hash of the dataset, the collection and how many chunks it held with
  a hash of their texts, `top_k`, the concurrency, what the RAG model and the judge were sent
  (temperature, effort or thinking level), the embedding model, the reranker and MMR. The
  Markdown report shows it, with the 95% Wilson interval of the accuracy and the hallucination
  rate: 48 right out of 50 is 86.5% to 98.9%. A diff warns when two runs searched different
  chunks: a collection deleted and seeded again with another chunker, or with documents added
  or changed.
- **Eval temperature.** A run answers at `AGENT_EVAL_RAG_TEMPERATURE` (0), not at the chat's
  0.3. Claude 4.7+ and Gemini 3+ take no temperature, so for them it changes nothing.
- **Judge cache.** The judge keeps its verdicts in `eval/cache/judge.jsonl`
  (`AGENT_EVAL_JUDGE_CACHE`, gitignored). The same answer to the same question, with the same
  judge model, settings and prompts, gets the verdict it got before, with no call.
  `--no-judge-cache` judges everything again.
- **Series.** `runner run --runs 5` runs the config five times and writes
  `series_<time>.md`: the mean, minimum, maximum and 95% interval of the mean (Student's t),
  each run, and the items whose verdict differed between runs. The thresholds apply to the
  mean.
- **Diff.** `runner diff` lists what differs between the two configs, and warns when the
  judge or the dataset changed or a run had uncommitted code. It lists the items that got
  worse and those that got better, with McNemar's exact p over them: 6 flips one way and none
  back is p = 0.031, the fewest that reads as a change at 0.05. It does not compare two runs
  over different datasets.
- **Judge consistency.** `runner rejudge eval/results/<run>.json --times 3` judges a saved
  run's answers three more times, with no cache, and writes `rejudge_<run>_<time>.md`: per
  check, the items whose verdicts all agree, and the share of new verdicts that differ from
  the run's. It reads the expected answers from the dataset the run read (`--dataset`,
  checked against the run's hash), and the chunks the judge read from the database: a report
  keeps their ids and a hash of their text, and an answer whose chunks are gone or changed is
  left out of the grounding check. It calls the judge's paid API.

Dataset: JSONL at `eval/golden/queries.jsonl`. Each in-scope item carries `required_facts` and
`optional_facts` (see [How the judge rules](#how-the-judge-rules)).

### Last full run

| Metric | Value |
|---|---|
| Answer accuracy (LLM judge) | 96% (48/50) |
| Unsupported-claim rate, in-scope items (faithfulness) | 7.3% (3/41) |
| Unsupported-claim rate, all items (the gates' hallucination rate) | 6% (3/50) |
| Retrieval precision@k / recall@k | 91.5% / 92.1% (40 of the 41 in-scope queries) |
| Out-of-scope queries refused by the RAG prompt | 9/9 |
| Single-turn RAG latency p50 / p95 | 5.9 s / 9.1 s |
| Generation cost (RAG model only) | $0.63 for the 50 queries, about $0.013 per answer |

Run `b2b0899a` on 2026-04-28, with Claude Sonnet 4.5 on Bedrock for generation. The full
report, including the five failing cases, is in
[`eval/reports/2026-04-28_b2b0899a.md`](eval/reports/2026-04-28_b2b0899a.md). Caveats:

- It used the original Spanish version of the seed corpus and the golden set. This repo ships
  the English translation, which is why the questions in that report are in Spanish; a run
  against this repo measures the translated set.
- It is one run of a stochastic pipeline. The seven full runs that day, made while tuning the
  engine, ranged from 80% to 96% accuracy.
- It predates this snapshot by three weeks and has not been re-run against it.
- It was judged by the earlier judge, which ruled before it reasoned and compared each answer
  with the expected one as a whole. The judge in this repo checks facts (see
  [How the judge rules](#how-the-judge-rules)), so its verdicts are not comparable with these,
  and it has not been run yet.
- The three unsupported claims are claims the retrieved context did not back, not necessarily
  false ones: #38's answer explains how SQL injection works, which the documents do not.
- The cost per answer is the Sonnet generation call alone. A chat turn also pays the
  Gemini Flash classifier, the query rewrite on follow-ups and the query embedding. The
  chat footer prices the generation too; the turn's trace carries the total
  (`whisperdocs.cost_usd`, see [Tracing](#tracing)).

That run averaged retrieval only over queries with a non-zero precision or recall, which
dropped the one in-scope query that retrieved no chunk from its expected source; over all 41
the figures are 89.3% and 89.8%. That query, #23, was mislabeled: the fact it asks about is in
two documents and the golden set listed only one. It now lists both, and
[`app/eval/metrics.py`](app/eval/metrics.py) averages over every in-scope query.

### A/B: measure reranker + MMR lift

```bash
RERANKER_ENABLED=false MMR_ENABLED=false  python -m app.eval.runner run --markdown
uv sync --frozen --extra dev --extra rerank   # list every extra: sync removes the rest
RERANKER_ENABLED=true  MMR_ENABLED=true   python -m app.eval.runner run --markdown
python -m app.eval.runner diff <baseline>.json <reranked>.json -o eval/results/reranker-delta.md
```

The diff report shows per-metric deltas, the items that got worse and better, and McNemar's p.

### Retrieval only, without LLM calls

```bash
python -m scripts.retrieval_bench --embeddings-cache eval/results/query-embeddings.json
```

Runs the vector leg, the full-text leg and the fused pipeline over the golden set and reports
how many questions each returns rows for, plus precision@k and recall@k. The only paid call is
one embedding per question, cached. Toggle `RERANKER_ENABLED` / `MMR_ENABLED` to A/B them for
free.

On the English golden set (k = 5, reranker and MMR off, the same query embeddings for each row):

| | Full-text rows (in-scope) | Hybrid precision@5 | Hybrid recall@5 |
|---|---|---|---|
| Before the full-text rework | 5 / 41 | 89.3% | 93.5% |
| After | 41 / 41 | 89.8% | 95.9% |
| Chunks by section (95 chunks, median 305 tokens, from 71 of 467) | 41 / 41 | 91.7% | 93.5% |

Before, the full-text leg ANDed every word of the question and almost never matched, so the
hybrid was vector-only in practice. Details, and why MMR at the default λ costs precision, in
[ADR 0003](docs/adr/0003-hybrid-retrieval-rrf.md).

Smaller chunks trade recall for precision. The vector leg's precision went from 89.8% to
95.6%, but on three questions whose answer spans two documents (#9, #28, #37) the main
document's extra chunks now fill the top 5 and push the second one out; #36 found its third.
Five chunks are also fewer tokens than before (about 1,500, from 2,300). At k = 8, about the
same tokens as five chunks before, the hybrid gives 85.7% precision and 97.2% recall. Whether
the change helps the answers is the full eval's to say.

---

## Tests

```bash
uv run pytest -q
# 794 passed, 84 skipped

# Database tests (full-text, HNSW, job cleanup) against a real Postgres, migrated with `alembic upgrade head`:
TEST_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5442/whisperdocs_test uv run pytest -q
# 878 passed
```

Layered: unit (`tests/agent/` — nodes against `FakeProvider` + mocked retriever), service (`tests/test_reranker.py`, `test_mmr.py`, `test_chunker.py`, `test_document_formats.py`, `test_retrieval_fusion.py`, `test_ingestion_offloop.py`, `test_ingestion_jobs.py`, `test_extraction.py`, `test_mcp_sync.py`), API (`tests/test_upload_limits.py`: the app over ASGI, database and embedder replaced), MCP (`tests/test_mcp.py`: an in-memory MCP client, and the stdio process; `test_mcp_http.py`: the key check in front of `/mcp`), database (`tests/test_retrieval_db.py`, `test_mcp_db.py`, `test_mcp_http_db.py`, `test_mcp_writes_db.py`, `test_mcp_sync_db.py`, `test_extraction_db.py` and one test in `test_ingestion_jobs.py`, skipped without `TEST_DATABASE_URL`). No test reaches a model or embedding API, even with credentials in `.env`: `tests/conftest.py` fails any test that resolves a host other than this machine or the test database.

CI (`.github/workflows/tests.yml`) runs, on every push to `main` and every pull request:
- `pre-commit run --all-files`: ruff, ruff format and file hygiene;
- gitleaks over the full history;
- pip-audit over every locked package;
- mypy (`app/agent`, `app/eval`, `app/mcp`, `app/connectors` and what they import);
- `alembic check`, after the migrations: the models match the schema they build;
- the full suite against a pgvector Postgres service, database tests included, with the
  coverage of `app/`: under 88% fails (90.6% on 2026-11-10; a run without a database measures
  about 87%, since the database tests are skipped). Locally: `uv run pytest --cov`. The
  default tracer does not see an MCP tool's code after its first `await` under the SDK's
  in-memory client, which undercounts the tools; `COVERAGE_CORE=sysmon` measures 92.3%.

Dependabot proposes weekly updates for Python, npm, GitHub Actions and Docker images.

---

## Project layout

```
app/
  agent/          LangGraph agent (nodes, graph, state, prompts, utils)
  api/v1/         FastAPI routers — agent, query, retrieve, documents,
                  conversations, notes, collections, eval, admin, webhooks,
                  health, metrics
  mcp/            MCP server (stdio, and /mcp over HTTP): tools, document
                  resources, scoped keys, the write tools
  connectors/     mcp_sync: another MCP server's resources into a collection
  services/       llm/ (providers + pricing), retriever, embedder, chunker,
                  reranker, mmr, retrieval_pipeline, ingestion, text_extractor,
                  rag, lang, whatsapp/, twilio_client; search, answering,
                  catalog, documents, extraction and audit, shared by the API
                  routes and the MCP tools
  models/         SQLAlchemy ORM
  schemas/        Pydantic
  eval/           Evaluation framework (runner, judge, evaluator, report, metrics)
alembic/          DB migrations
fixtures/seed-docs/   Curated corpus for the eval suite
frontend/         React + TanStack Router + shadcn/ui
tests/            Pytest
docs/adr/         Architecture Decision Records
```

---

## License

MIT
