# Architecture

Reference document for the WhisperDocs system. Companion to [`README.md`](README.md).
Decision records under [`docs/adr/`](docs/adr/) explain *why* the major shapes
were chosen.

---

## Component diagram

```mermaid
graph TB
  subgraph Client
    UI[React + TanStack Router]
    WApp[WhatsApp]
  end

  subgraph API[FastAPI]
    Agent["/api/v1/agent/chat<br/>+ /chat/stream"]
    Query["/api/v1/query"]
    Docs["/api/v1/documents"]
    Eval["/api/v1/eval"]
    Webhook["/api/v1/webhooks/whatsapp"]
    Health["/api/v1/health"]
    Metrics["/api/v1/metrics"]
  end

  subgraph AgentCore[LangGraph Agent]
    Graph[StateGraph]
    Compaction[check_compaction]
    Summarize[summarize]
    Classifier[decide_action]
    Search[search_docs]
    Clarify[ask_clarification]
    Direct[answer_direct]
    OutOfScope[out_of_scope]
    Checkpointer[(AsyncPostgresSaver)]
  end

  subgraph Services
    Retrieval[Retrieval Pipeline]
    Hybrid[VectorRetriever<br/>vector + BM25 + RRF]
    Reranker[Cross-encoder Reranker]
    MMR[MMR diversity]
    Embedder[OpenAI Embeddings]
    LLM[LLM Provider Abstraction]
    Bedrock[Bedrock Claude]
    Gemini[Gemini 2.5 Flash]
    Ingestion[Async Ingestion Worker]
    Pricing[pricing.py]
  end

  subgraph Storage
    PG[(Postgres + pgvector)]
    Files[Filesystem / blob]
  end

  UI -->|HTTP / SSE| API
  WApp -->|webhook| Webhook
  Agent --> Graph
  Webhook --> Graph
  Graph --> Compaction --> Summarize
  Compaction --> Classifier
  Classifier --> Search
  Classifier --> Clarify
  Classifier --> Direct
  Classifier --> OutOfScope
  Search --> Retrieval
  Retrieval --> Hybrid
  Hybrid --> Reranker
  Reranker --> MMR
  Search --> LLM
  Clarify --> LLM
  Direct --> LLM
  Summarize --> LLM
  LLM --> Bedrock
  LLM --> Gemini
  Hybrid --> PG
  Embedder --> PG
  Docs --> Ingestion
  Ingestion --> Embedder
  Ingestion --> PG
  Graph --> Checkpointer
  Checkpointer --> PG
  Eval --> LLM
  Eval --> Retrieval
  Search --> Pricing
  Query --> Pricing
```

---

## Sequence: agent chat (streaming)

```mermaid
sequenceDiagram
  autonumber
  participant U as User (UI)
  participant API as FastAPI
  participant G as LangGraph Agent
  participant LLM as Provider (Bedrock / Gemini)
  participant DB as Postgres + pgvector

  U->>API: POST /agent/chat/stream
  API->>DB: load checkpoint (thread_id)
  API->>G: astream_events(message)
  G->>G: check_compaction
  alt threshold exceeded
    G->>LLM: summarize history
    LLM-->>G: summary
  end
  G->>LLM: classifier (Gemini)
  LLM-->>G: AgentAction { search_docs | ask_clarification | answer_direct | out_of_scope }
  alt action == search_docs
    G->>LLM: rewrite_query (if not self-contained)
    G->>DB: hybrid_retrieve (vector + BM25 + RRF)
    DB-->>G: top_k chunks
    opt reranker enabled
      G->>G: cross-encoder rerank → MMR
    end
    G->>LLM: generate_stream(system, history + chunks)
    loop per token
      LLM-->>G: delta
      G-->>API: dispatch_custom_event("agent_token", delta)
      API-->>U: SSE event: token
    end
    LLM-->>G: final
  else action == answer_direct / ask_clarification
    G->>LLM: generate_stream
    LLM-->>G: tokens (streamed same way)
  else action == out_of_scope
    G-->>API: canned refusal (no LLM call)
  end
  G->>DB: persist checkpoint + QueryLog
  G-->>API: final state
  API-->>U: SSE event: final { answer, citations, metadata }
```

---

## Sequence: async ingestion

```mermaid
sequenceDiagram
  autonumber
  participant U as User (UI)
  participant API as FastAPI
  participant W as Ingestion Worker
  participant E as Embedder (OpenAI)
  participant DB as Postgres

  U->>API: POST /documents/upload-async (file)
  API->>DB: INSERT IngestionJob (pending)
  API-->>U: 202 { job_id }
  API->>W: spawn asyncio task
  par stream subscribe
    U->>API: GET /documents/jobs/{id}/stream
    loop poll every 500ms
      API->>DB: SELECT IngestionJob
      API-->>U: SSE event: progress { status, progress, processed_chunks }
    end
  and worker pipeline
    W->>DB: status = extracting
    W->>W: extract_text
    W->>DB: status = chunking
    W->>W: TextChunker.chunk
    W->>DB: dedup hash check
    W->>DB: status = embedding, total_chunks=N
    loop per batch (32)
      W->>E: embed_batch
      E-->>W: vectors
      W->>DB: progress += batch_pct
    end
    W->>DB: status = persisting → INSERT Document + Chunks
    W->>DB: status = complete
  end
  API-->>U: SSE event: complete { document_id }
```

---

## Data model

```mermaid
erDiagram
  Collection ||--o{ Document : contains
  Document ||--o{ Chunk : "has many"
  Document ||--o{ IngestionJob : "tracked by"
  Collection ||--o{ IngestionJob : "scoped to"
  Conversation ||--o{ SavedNote : "pins"
  Collection ||--o{ Conversation : "default scope"
  Conversation ||--o{ QueryLog : logs
  Collection ||--o{ QueryLog : "scoped to"

  Collection {
    int id PK
    string name UK
    text description
    timestamp created_at
  }

  Document {
    int id PK
    int collection_id FK
    string title
    string filename
    string file_type
    text raw_text
    int chunk_count
    int total_tokens
    string language
    timestamp created_at
  }

  Chunk {
    int id PK
    int document_id FK
    int chunk_index
    text text
    int token_count
    string text_hash
    vector embedding "pgvector(1536) HNSW"
    tsvector tsv "BM25 generated col"
  }

  IngestionJob {
    int id PK
    int collection_id FK
    int document_id FK
    string filename
    string status
    int progress
    int total_chunks
    int processed_chunks
    text error_message
    timestamp created_at
    timestamp started_at
    timestamp completed_at
  }

  Conversation {
    int id PK
    string thread_id UK
    string user_number
    string channel
    int collection_id FK
    bool is_saved
    timestamp last_active_at
    timestamp created_at
  }

  QueryLog {
    int id PK
    text question
    text answer
    int collection_id FK
    int chunks_retrieved
    string model_used
    int tokens_input
    int tokens_output
    float cost_usd
    int latency_ms
    string session_id
    string agent_action
    timestamp created_at
  }

  SavedNote {
    int id PK
    int conversation_id FK
    int collection_id FK
    text source_message_content
    text user_note
    timestamp created_at
  }
```

LangGraph checkpointer state lives in dedicated tables managed by
`langgraph-checkpoint-postgres` (not modeled here).

---

## State contract

`AgentState` (`app/agent/state.py`) is the central data structure flowing
between graph nodes. Splits into two lifetimes:

- **Persistent** (survive across turns via checkpointer): `messages`,
  `summary`, `pending_action`, `collection_id`, `metadata`, `user_number`,
  `channel`, `notes_pinned`, `turn_count`.
- **Transient** (reset by `check_compaction` each turn): `retrieved_chunks`,
  `top_k`.

Each node is a pure async function `(AgentState) -> dict` returning only the
fields it modifies. The graph wires nodes in `app/agent/graph.py`; nodes never
import each other directly. See `.claude/rules/agent.md` for the full contract.

---

## Retrieval pipeline

`app/services/retrieval_pipeline.py::retrieve()` composes:

1. **Hybrid fetch** — `VectorRetriever.hybrid_retrieve(query_text, query_embedding, ...)`
   - Vector search via cosine distance + HNSW index on `chunks.embedding`
   - BM25 via `ts_rank_cd(c.tsv, websearch_to_tsquery(...))`
   - Fused with Reciprocal Rank Fusion (`1 / (k + rank)`, `k=60`)
2. **Cross-encoder rerank** (optional, `RERANKER_ENABLED=true`)
   - `BAAI/bge-reranker-base` via `sentence-transformers`
   - Scores `(query, chunk_text)` pairs, sorts by score, normalizes to 0-1
3. **MMR diversity** (optional, `MMR_ENABLED=true`)
   - Document-level penalty: `λ * relevance(c) − (1−λ) * I[doc(c) ∈ selected_docs]`
   - λ=1.0 → pure relevance; λ=0.0 → pure diversity

Pool sizing: when reranking is active, `hybrid_retrieve` fetches
`max(top_k, RERANKER_CANDIDATE_POOL=20)` candidates. Without reranking,
`top_k` directly.

---

## Provider abstraction

```python
class LLMProvider(Protocol):
    async def generate(...) -> LLMResponse: ...
    async def generate_structured(...) -> tuple[T, LLMResponse]: ...
    def generate_stream(...) -> AsyncIterator[LLMStreamDelta]: ...
```

`app/services/llm/__init__.py::llm_service_for(node_alias)` routes by config:

| Alias | Default model | Provider |
|-------|---------------|----------|
| `classifier` | `gemini-2.5-flash` | Gemini |
| `query_rewrite` | `gemini-2.5-flash` | Gemini |
| `clarification` | `gemini-2.5-flash` | Gemini |
| `answer_direct` | `gemini-2.5-flash` | Gemini |
| `rag` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Bedrock |
| `summarize` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock |
| `eval_judge` | `gemini-2.5-flash` | Gemini |

Override per node via `AGENT_MODEL_<ALIAS>` env vars.

---

## Eval framework

`app/eval/` is a self-contained subsystem:

- `schemas.py` — `TestItem`, `EvalResult`, `AggregateMetrics`, `EvalReport`
- `runner.py` — CLI: `run` and `diff` subcommands, JSONL or JSON dataset loaders
- `evaluator.py` — orchestrator (loads dataset, embeds, retrieves, generates, judges)
- `judges.py` — LLM-as-judge for correctness + hallucination
- `metrics.py` — aggregation: precision, recall@k, p50/p95, cost via shared `pricing.py`
- `report.py` — Markdown rendering + diff between runs

Dataset format (`eval/golden/queries.jsonl`):

```jsonl
{"id": 1, "question": "...", "expected_answer": "...", "difficulty": "medium", "category": "in_scope", "expected_sources": ["foo.md"]}
```

The same retrieval pipeline (`retrieval_pipeline.retrieve`) is used in eval and
production — toggling reranker / MMR via env vars produces an apples-to-apples
A/B run.

---

## Observability

- **Logs** — `structlog` with JSON output in production (`LOG_JSON=1`),
  console-pretty otherwise. `merge_contextvars` ready for `request_id` /
  `trace_id` propagation.
- **Metrics** — Prometheus `/metrics` endpoint:
  - `whisperdocs_agent_turn_duration_seconds{node}`
  - `whisperdocs_llm_tokens_total{provider,model,direction}`
  - `whisperdocs_retrieval_chunks{collection}`
  - `whisperdocs_retrieval_top_relevance` (histogram)
  - `whisperdocs_errors_total{type}`
- **Cost tracking** — `query_logs.cost_usd` populated per turn via `pricing.estimate_cost_usd(model, tokens_in, tokens_out)`.
- **Health** — `/health` with `?deep=true` checks DB ping + pgvector extension + embedder reachability.

---

## Testing strategy

- **Unit** — Each agent node tested in isolation with `FakeProvider` and a
  mocked retrieval pipeline (`tests/agent/conftest.py`). Streaming nodes get a
  `generate_stream` stub that yields the canned response.
- **Service** — `test_chunker.py`, `test_reranker.py`, `test_mmr.py` exercise
  pure-function services without DB.
- **Integration (TODO)** — `pytest-postgresql` / testcontainers wiring not yet
  in CI.
- **E2E** — Eval suite acts as integration: ingest fixture corpus, query, judge.

`tests/agent/conftest.py::mock_retriever` patches both
`retrieval_pipeline.retrieve` *and* the rebound names in importing modules so
either entry point sees the same stub data.

---

## Decision records

See [`docs/adr/`](docs/adr/) for short rationale notes:

- [0001 — LangGraph over chained LCEL](docs/adr/0001-langgraph-over-lcel.md)
- [0002 — pgvector over a dedicated vector DB](docs/adr/0002-pgvector-over-dedicated-vectordb.md)
- [0003 — Hybrid retrieval with RRF](docs/adr/0003-hybrid-retrieval-rrf.md)
- [0004 — Dual LLM provider routing](docs/adr/0004-dual-llm-provider-routing.md)
