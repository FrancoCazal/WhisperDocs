# WhisperDocs

[![Tests](https://github.com/FrancoCazal/WhisperDocs/actions/workflows/tests.yml/badge.svg)](https://github.com/FrancoCazal/WhisperDocs/actions/workflows/tests.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A self-hosted RAG engine with an intent-routed agent: it answers questions about your own documents with citations, and measures how good and how expensive those answers are. For teams that need answers from their own documentation, in a web chat or over WhatsApp, with numbers on quality and cost. What sets it apart: an eval suite with diffs between runs, the USD cost of each RAG answer, and a model chosen per graph node.

> **Portfolio project — runs locally via Docker. No hosted deployment.** Clone, fill `.env`, `docker compose up`.

> **Published as a snapshot.** WhisperDocs was built between April and May 2026 and pushed here in a single commit, so this repository does not carry the development history. The design decisions are recorded in [`docs/adr/`](docs/adr/).

![System overview dashboard](media/whisperdocs1.png)

**Stack** · FastAPI · PostgreSQL + pgvector · LangGraph · Gemini + AWS Bedrock (Claude) · OpenAI Embeddings · Twilio · React + TanStack Router · structlog · Prometheus

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for component diagrams, sequence flows, and data model. Decision records live under [`docs/adr/`](docs/adr/).

---

## Highlights

- **Intent-routed RAG agent, single pass (LangGraph)** — a classifier sends each turn down one path: search, clarify, direct answer or out-of-scope. Search rewrites the question, retrieves once and generates; there is no loop or second retrieval, by design. 7 nodes in an explicit state graph (compaction and summary keep long threads bounded), each testable in isolation; `AsyncPostgresSaver` handles persistence.
- **Hybrid retrieval** — vector cosine via pgvector + Postgres full-text search (OR over the question's stemmed terms, per-document language, accent-insensitive, ranked by `ts_rank_cd`), fused with Reciprocal Rank Fusion. Optional cross-encoder rerank (`bge-reranker-base`) + document-level MMR for diversity.
- **Multi-provider routing** — Gemini 3.8 Flash for cheap hot-path nodes (classifier, query rewrite, direct answers); AWS Bedrock Claude for grounded RAG generation. Provider auto-detected from model ID; startup fails on a model ID that can't be routed.
- **Token-level SSE streaming** — `astream_events` v2 + custom event dispatch. Frontend renders tokens as they arrive, with progress markers per node.
- **Async ingestion with progress** — `IngestionJob` row + asyncio worker. Per-stage progress (extracting → chunking → embedding → persisting). Frontend bar updates over SSE. Parsing runs off the event loop, and a job cut short by a restart ends as `failed`, not stuck.
- **Eval suite (50-query golden set)** — single-turn retrieval + generation path. LLM-as-judge for correctness + hallucination, retrieval precision/recall@k, latency p50/p95, generation cost USD. Diff between runs for regression detection.
- **Cost tracking** — per-call USD via `pricing.py`, persisted on `query_logs`. Single source for runner, API, dashboard.
- **Structured logging + metrics** — `structlog` JSON output, Prometheus `/metrics` (turn duration, tokens, retrieval relevance, errors).
- **WhatsApp integration** — full agent pipeline over Twilio webhooks.
- **MCP server** — the knowledge base for Claude Code, Claude Desktop and Cursor:
  search, answers with citations, the lists and a resource per document, over
  stdio. Each tool calls the same service as its REST route.

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

---

## Environment

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5435/whisperdocs
DB_PORT=5435                     # the compose database's port on the host; change it if 5435 is taken
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
| `GET` | `/api/v1/health` | Health check (`?deep=true` for full probe and each model role's state) |
| `GET` | `/api/v1/metrics` | Prometheus metrics |

Uploads accept PDF, Markdown and plain text, up to 10 MB per file. Batches
take up to 20 files and 50 MB in total. A larger body is rejected with 413
before it is parsed. A `.pdf` must carry the PDF header, a `.md` or `.txt`
must not contain binary data, and file names and titles are capped at 300
characters. List endpoints take `limit` between 1 and 200.

Full reference at `/docs` (FastAPI auto-generated OpenAPI). Every operation has a
fixed `operationId` (`retrieve_chunks`, `answer_question`, `list_collections`, ...),
so a client generated from the schema, or an agent's tool list, keeps its names;
`tests/test_openapi.py` lists them all.

---

## MCP server

WhisperDocs is also an MCP server, so Claude Code, Claude Desktop, Cursor or
any other MCP client can use the knowledge base. It runs over stdio, against
the compose Postgres, and no tool changes the knowledge base.

| Tool | What it returns |
|------|-----------------|
| `search_documents` | The passages for a query, with their text and document, and nothing generated: the client answers with its own model. Scoped by `collection_id` or `document_id`; `top_k` from 1 to 20. |
| `ask_documents` | WhisperDocs' own answer, with its citations: `[n]` in the answer is `citations[n-1]`. It pays for a model call and is saved to the query history. |
| `list_collections` | The collections, with their document, chunk and token totals. |
| `list_documents` | A page of documents, each with its resource URI. |

Each document is also a resource, `whisperdocs://documents/{id}`, holding its
full text. Each tool calls the service its REST route calls (`/retrieve`,
`/query`, the lists), so both return the same thing. The server speaks the
current MCP spec (2026-07-28) and the older `initialize` handshake that many
clients still use. [ADR 0005](docs/adr/0005-mcp-server.md) has the design.

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
`DATABASE_URL` reaches the database (`localhost:5435` with the compose
defaults).

- **No auth over stdio.** The client starts the process on your machine, and
  the process has the database and keys the API has. Serving MCP over HTTP
  waits for scoped keys.
- **Its own process.** The API's rate limits and its model status do not see
  the MCP server's calls; the server logs to stderr, which the client keeps
  as its MCP log.

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
- **Not covered.** There are no user accounts or roles: whoever has the key
  can read and delete every document and thread. The API serves no TLS, so
  put a TLS proxy in front of it. Documents and questions go to the
  configured providers: OpenAI for embeddings, Vertex AI or Bedrock for
  generation.

---

## Eval suite

Reproducible evaluation against a 50-query golden set covering all seed fixtures.

```bash
# Seed the eval corpus (idempotent).
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

Metrics: accuracy, hallucination rate, retrieval precision@k and recall@k (averaged over the
in-scope queries), latency avg/p50/p95, and generation cost in USD (RAG model only; judge calls
and embeddings are not included). Bucketed by difficulty and category. The run aborts with exit
code 2 when more than 10% of the judge calls fail (`AGENT_EVAL_MAX_JUDGE_ERROR_RATE`), instead of
scoring those items as failures.

Dataset: JSONL at `eval/golden/queries.jsonl`.

### Last full run

| Metric | Value |
|---|---|
| Answer accuracy (LLM judge) | 96% (48/50) |
| Hallucination rate | 6% (3/50) |
| Retrieval precision@k / recall@k | 91.5% / 92.1% (40 of the 41 in-scope queries) |
| Out-of-scope queries refused by the RAG prompt | 9/9 |
| Single-turn RAG latency p50 / p95 | 5.9 s / 9.1 s |
| Generation cost (RAG model only) | $0.63 for the 50 queries |

Run `b2b0899a` on 2026-04-28, with Claude Sonnet 4.5 on Bedrock for generation. The full
report, including the five failing cases, is in
[`eval/reports/2026-04-28_b2b0899a.md`](eval/reports/2026-04-28_b2b0899a.md). Three caveats:

- It used the original Spanish version of the seed corpus and the golden set. This repo ships
  the English translation, which is why the questions in that report are in Spanish; a run
  against this repo measures the translated set.
- It is one run of a stochastic pipeline. The seven full runs that day, made while tuning the
  engine, ranged from 80% to 96% accuracy.
- It predates this snapshot by three weeks and has not been re-run against it.

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

The diff report shows per-metric deltas and per-item regressions.

### Retrieval only, without LLM calls

```bash
python -m scripts.retrieval_bench --embeddings-cache eval/results/query-embeddings.json
```

Runs the vector leg, the full-text leg and the fused pipeline over the golden set and reports
how many questions each returns rows for, plus precision@k and recall@k. The only paid call is
one embedding per question, cached. Toggle `RERANKER_ENABLED` / `MMR_ENABLED` to A/B them for
free.

On the English golden set (k = 5, reranker and MMR off):

| | Full-text rows (in-scope) | Hybrid precision@5 | Hybrid recall@5 |
|---|---|---|---|
| Before the full-text rework | 5 / 41 | 89.3% | 93.5% |
| After | 41 / 41 | 89.8% | 95.9% |

Before, the full-text leg ANDed every word of the question and almost never matched, so the
hybrid was vector-only in practice. Details, and why MMR at the default λ costs precision, in
[ADR 0003](docs/adr/0003-hybrid-retrieval-rrf.md).

---

## Tests

```bash
uv run pytest -q
# 150 passed, 9 skipped

# Database tests (full-text, HNSW, job cleanup) against a real Postgres, migrated with `alembic upgrade head`:
TEST_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5435/whisperdocs_test uv run pytest -q
# 159 passed
```

Layered: unit (`tests/agent/` — nodes against `FakeProvider` + mocked retriever), service (`tests/test_reranker.py`, `test_mmr.py`, `test_chunker.py`, `test_retrieval_fusion.py`, `test_ingestion_offloop.py`, `test_ingestion_jobs.py`), API (`tests/test_upload_limits.py`: the app over ASGI, database and embedder replaced), MCP (`tests/test_mcp.py`: an in-memory MCP client, and the stdio process), database (`tests/test_retrieval_db.py`, `test_mcp_db.py` and one test in `test_ingestion_jobs.py`, skipped without `TEST_DATABASE_URL`). No test reaches a model or embedding API, even with credentials in `.env`: `tests/conftest.py` fails any test that resolves a host other than this machine or the test database.

CI (`.github/workflows/tests.yml`) runs, on every push to `main` and every pull request:
- `pre-commit run --all-files`: ruff, ruff format and file hygiene;
- gitleaks over the full history;
- pip-audit over every locked package;
- mypy (`app/agent`, `app/eval`, `app/mcp` and what they import);
- the full suite against a pgvector Postgres service, database tests included.

Dependabot proposes weekly updates for Python, npm, GitHub Actions and Docker images.

---

## Project layout

```
app/
  agent/          LangGraph agent (nodes, graph, state, prompts, utils)
  api/v1/         FastAPI routers — agent, query, retrieve, documents,
                  conversations, notes, collections, eval, admin, webhooks,
                  health, metrics
  mcp/            MCP server (stdio): tools and document resources
  services/       llm/ (providers + pricing), retriever, embedder, chunker,
                  reranker, mmr, retrieval_pipeline, ingestion, text_extractor,
                  rag, lang, whatsapp/, twilio_client; search, answering and
                  catalog, shared by the API routes and the MCP tools
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
