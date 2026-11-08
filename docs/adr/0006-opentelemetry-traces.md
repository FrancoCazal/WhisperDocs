# ADR 0006 — OpenTelemetry traces with the GenAI conventions

**Status:** accepted
**Date:** 2026-11

## Context

A turn makes up to four model calls (classifier, query rewrite, answer,
summary), an embedding request and a retrieval. Prometheus counts them in
aggregate and the logs say what failed, but nothing showed one turn as a
whole: which node took the time, what each call cost, or which log lines and
which `query_logs` row belong to it. The cost shown for a turn was the
answer's only; the classifier and the summary were never counted (audit A5).

The OpenTelemetry GenAI semantic conventions name what a model call,
an embedding, a retrieval and an agent or workflow invocation record. They
moved to their own repository, open-telemetry/semantic-conventions-genai,
and are still in **Development**: names can change, and there are no tagged
releases yet. Langfuse, Phoenix and the OpenTelemetry backends read them.

FastAPI 0.162 emits OpenTelemetry spans of its own for each request when a
TracerProvider is installed.

## Decision

- **Our own instrumentation, in one module (`app/telemetry.py`).** Every
  model call already goes through `TrackedProvider`, every embedding through
  `OpenAIEmbeddingProvider.embed_batch`, every search through
  `retrieval_pipeline.retrieve`, and the graph's nodes are wired in one
  place. A span at each of those points covers the turn, with no
  instrumentation library.
- **The conventions, pinned to a date.** The attribute names are those of
  the conventions' repository on 2026-11-06, each spelled once in
  `app/telemetry.py`. The turn is `invoke_workflow`, not `invoke_agent`: the
  graph decides the control flow, not a model, and the conventions name
  LangGraph's `StateGraph` as the example. A model call is `chat {model}`,
  CLIENT, with its provider (`aws.bedrock`, `gcp.vertex_ai`), `max_tokens`,
  the temperature and the effort only when they are sent, token counts and
  `error.type`. The thread is `gen_ai.conversation.id`.
- **No content.** No span records a prompt, an answer, a query or a chunk's
  text, which the conventions make opt-in, nor an exception's message:
  `error.type` is the `classify_error` kind (`timeout`, `billing`, ...),
  because a provider's message can quote the prompt.
- **The cost is ours.** Each call's cost comes from `pricing.py`, the table
  the API and the eval use. The turn's span adds up the tokens and cost of
  every call made inside it (a context variable the graph's tasks share), so
  the classifier and the summary count. Those totals are named
  `whisperdocs.input_tokens`, `whisperdocs.output_tokens` and
  `whisperdocs.cost_usd`, outside the conventions: Langfuse adds up
  `gen_ai.usage.*` over every span of a trace, and with the turn carrying
  them it showed twice the tokens. `query_logs.cost_usd` stays the answer's
  cost, the same number the chat shows.
- **Trace ids always, export on request.** A TracerProvider is always
  installed, so each turn has a trace id for the logs (`trace_id`,
  `span_id`) and for `query_logs.trace_id`. Spans leave the process only when
  `OTEL_EXPORTER_OTLP_ENDPOINT` is set, over OTLP/HTTP.
- **FastAPI's spans are the root.** Its span of the request is the trace's
  root and the turn nests under it; an incoming `traceparent` is honored.
  Its own exporters, metrics, logs and per-dependency spans stay off.
- **Langfuse as the local viewer.** A compose profile, `tracing`, runs
  Langfuse 4.76 self-hosted, with its project and keys created on the first
  start, so traces show with two lines in `.env`.

## Why

- **One trace per turn answers the questions that came up.** Which node is
  slow, what a turn costs, which call failed, which row and log lines go
  with it.
- **The conventions keep the backend open.** The same spans read in
  Langfuse, Phoenix, Jaeger or any OTLP backend; changing the viewer is an
  environment variable.
- **The chokepoints already existed.** `TrackedProvider` was added for the
  model status (BE-14); tracing is a few lines at the same places, and the
  tests check each span with an in-memory exporter.

## Alternatives considered

- **An instrumentation library** (OpenLLMetry, OpenInference, the Langfuse
  SDK's LangChain callback). They instrument LangChain's calls, not ours: the
  providers call `ChatBedrockConverse` and `ChatGoogleGenerativeAI` through
  LangChain, so the spans would show LangChain's internals, record message
  content by default, and price the models from their own tables.
- **LangSmith.** Hosted, LangChain-specific, and outside the OpenTelemetry
  backends.
- **Phoenix as the local viewer.** One container instead of six, and it
  reads OTLP. Langfuse was chosen because the roadmap names it, it maps the
  GenAI attributes (model, usage, cost, session) into its own views, and it
  is the one users ask for; Phoenix works with the same two variables.
- **`invoke_agent` for the turn.** The conventions keep it for an agent
  whose model decides the control flow.

## Consequences

- When the conventions rename an attribute, `app/telemetry.py` changes, and
  the test that lists a model call's attributes fails until it does.
- The Langfuse profile adds six containers (web, worker, Postgres,
  ClickHouse, Redis, MinIO). It is opt-in, and its credentials are local-only
  values in the compose file.
- An answer's trace and its `query_logs` row share an id, so the stats can
  link to the trace once the frontend shows it.
