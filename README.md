# WhisperDocs

[![Tests](https://github.com/FrancoCazal/WhisperDocs/actions/workflows/tests.yml/badge.svg)](https://github.com/FrancoCazal/WhisperDocs/actions/workflows/tests.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Agentic RAG over your own documents — chat UI, WhatsApp delivery, hybrid retrieval, multi-provider LLMs, real eval suite.

> **Portfolio project — runs locally via Docker. No hosted deployment.** Clone, fill `.env`, `docker compose up`.

> **Published as a snapshot.** WhisperDocs was built between April and May 2026 and pushed here in a single commit, so this repository does not carry the development history. The design decisions are recorded in [`docs/adr/`](docs/adr/).

![System overview dashboard](media/whisperdocs1.png)

**Stack** · FastAPI · PostgreSQL + pgvector · LangGraph · Gemini + AWS Bedrock (Claude) · OpenAI Embeddings · Twilio · React + TanStack Router · structlog · Prometheus

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for component diagrams, sequence flows, and data model. Decision records live under [`docs/adr/`](docs/adr/).

---

## Highlights

- **Agentic RAG with LangGraph** — explicit state graph, 7 nodes (compaction, summarize, classifier, search, clarify, direct, out-of-scope). Each node testable in isolation; `AsyncPostgresSaver` handles persistence.
- **Hybrid retrieval** — vector cosine via pgvector + Postgres full-text search (OR over the question's stemmed terms, per-document language, accent-insensitive, ranked by `ts_rank_cd`), fused with Reciprocal Rank Fusion. Optional cross-encoder rerank (`bge-reranker-base`) + document-level MMR for diversity.
- **Multi-provider routing** — Gemini 3.8 Flash for cheap hot-path nodes (classifier, query rewrite, direct answers); AWS Bedrock Claude for grounded RAG generation. Provider auto-detected from model ID; startup fails on a model ID that can't be routed.
- **Token-level SSE streaming** — `astream_events` v2 + custom event dispatch. Frontend renders tokens as they arrive, with progress markers per node.
- **Async ingestion with progress** — `IngestionJob` row + asyncio worker. Per-stage progress (extracting → chunking → embedding → persisting). Frontend bar updates over SSE. Parsing runs off the event loop, and a job cut short by a restart ends as `failed`, not stuck.
- **Eval suite (50-query golden set)** — single-turn retrieval + generation path. LLM-as-judge for correctness + hallucination, retrieval precision/recall@k, latency p50/p95, generation cost USD. Diff between runs for regression detection.
- **Cost tracking** — per-call USD via `pricing.py`, persisted on `query_logs`. Single source for runner, API, dashboard.
- **Structured logging + metrics** — `structlog` JSON output, Prometheus `/metrics` (turn duration, tokens, retrieval relevance, errors).
- **WhatsApp integration** — full agent pipeline over Twilio webhooks.

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
OPENAI_API_KEY=sk-...

# Generation providers
GCP_PROJECT=...                  # Gemini via Vertex AI; auth via `gcloud auth application-default login`
GCP_LOCATION=global              # gemini-3.8-flash is served from global, us or eu only
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Optional
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
| `GET` | `/api/v1/query/history` | Paginated query log |
| `POST` | `/api/v1/agent/chat` | Multi-turn agent chat |
| `POST` | `/api/v1/agent/chat/stream` | SSE: node + token events |
| `GET` | `/api/v1/conversations` | List conversations |
| `GET` | `/api/v1/conversations/{thread_id}/messages` | Conversation history |
| `PATCH` | `/api/v1/conversations/{thread_id}` | Update conversation (e.g. `is_saved`) |
| `DELETE` | `/api/v1/conversations/{thread_id}` | Delete conversation |
| `GET` | `/api/v1/notes` | List saved notes |
| `DELETE` | `/api/v1/notes/{note_id}` | Delete saved note |

Uploads accept PDF, Markdown and plain text, up to 10 MB per file. Batches
take up to 20 files and 50 MB in total. A larger body is rejected with 413
before it is parsed. A `.pdf` must carry the PDF header, a `.md` or `.txt`
must not contain binary data, and file names and titles are capped at 300
characters. List endpoints take `limit` between 1 and 200.
| `POST` | `/api/v1/webhook/whatsapp` | Twilio inbound webhook |
| `POST` | `/api/v1/webhook/whatsapp/status` | Twilio delivery-status callback |
| `POST` | `/api/v1/eval/run` | Eval suite |
| `GET` | `/api/v1/eval/results` | List past eval reports |
| `GET` | `/api/v1/admin/settings` | Runtime settings snapshot |
| `GET` | `/api/v1/health` | Health check (`?deep=true` for full probe) |
| `GET` | `/api/v1/metrics` | Prometheus metrics |

Full reference at `/docs` (FastAPI auto-generated OpenAPI).

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

Layered: unit (`tests/agent/` — nodes against `FakeProvider` + mocked retriever), service (`tests/test_reranker.py`, `test_mmr.py`, `test_chunker.py`, `test_retrieval_fusion.py`, `test_ingestion_offloop.py`, `test_ingestion_jobs.py`), API (`tests/test_upload_limits.py`: the app over ASGI, database and embedder replaced), database (`tests/test_retrieval_db.py` and one test in `test_ingestion_jobs.py`, skipped without `TEST_DATABASE_URL`).

---

## Project layout

```
app/
  agent/          LangGraph agent (nodes, graph, state, prompts, utils)
  api/v1/         FastAPI routers — agent, query, documents, conversations,
                  notes, collections, eval, admin, webhooks, health, metrics
  services/       llm/ (providers + pricing), retriever, embedder, chunker,
                  reranker, mmr, retrieval_pipeline, ingestion, text_extractor,
                  rag, lang, whatsapp/, twilio_client
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
