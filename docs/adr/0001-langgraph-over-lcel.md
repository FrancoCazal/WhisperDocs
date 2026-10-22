# ADR 0001 — LangGraph over chained LCEL

**Status:** accepted
**Date:** 2026-04

## Context

The agent loop is "classify → branch → maybe retrieve → generate → maybe
summarize for next turn → checkpoint". Two reasonable framings:

1. **LCEL chain** — pipe `Runnable`s (`prompt | llm | parser`), branch with
   `RunnableBranch`. Lightweight; no graph DSL.
2. **LangGraph** — explicit `StateGraph` with named nodes, conditional edges,
   typed state, and a built-in checkpointer.

## Decision

Use LangGraph.

## Why

- **State is non-trivial.** Persistent fields (`messages`, `summary`,
  `pending_action`, `collection_id`, `metadata`, `turn_count`) survive turns;
  transient fields (`retrieved_chunks`, `top_k`) reset each turn. LCEL's
  one-shot input/output shape forces ad-hoc dict munging at every step.
- **Multi-turn memory is core, not optional.** `AsyncPostgresSaver`
  ("checkpointer") handles persistence per `thread_id` for free in LangGraph.
  In LCEL we'd hand-roll it.
- **Branching beyond simple conditionals.** `decide_action` routes to one of
  three nodes, each writing different `metadata` shapes. Conditional edges +
  TypedDict state are clearer than nested `RunnableBranch`.
- **Streaming events.** `astream_events("v2")` gives node-level lifecycle
  events out of the box, which is what powers our SSE endpoint
  (`node_start`, `node_end`, plus custom `agent_token` events). Faking the same
  semantics over LCEL is doable but bespoke.
- **Testability.** Each node is a pure async function. Nothing in LCEL prevents
  this — but the framework actively encourages it; LCEL invites you to compose
  pipelines that span multiple LLM calls inside a single chain object.

## Trade-offs

- More framework surface area; LangGraph is newer than LCEL and has more
  churn. Mitigated by keeping graph wiring in a single file (`app/agent/graph.py`)
  and treating nodes as plain functions.
- The `Annotated[..., reducer]` magic for state merging is non-obvious; we
  document it in `app/agent/state.py`.

## Out of scope

`Annotated[list[BaseMessage], add_messages]` vs. `operator.add` reducer choices
are documented in code, not here.
