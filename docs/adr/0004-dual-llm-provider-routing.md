# ADR 0004 — Dual LLM provider routing

**Status:** accepted
**Date:** 2026-04

## Context

Different agent nodes have different cost/quality requirements:

- **Classifier** runs on every turn. Latency-sensitive, low complexity (3-way
  intent classification with structured output).
- **RAG generation** runs on retrieval-bound turns. Quality-sensitive,
  long context (~5K-15K tokens with retrieved chunks), needs strong grounding
  and faithful citation behaviour.
- **Summarize** runs occasionally for compaction. Quality-sensitive,
  moderately latency-tolerant.
- **Direct answer / clarification** runs for chitchat / ambiguity. Cheap,
  non-grounded.

Single-provider strategies waste money or quality on the mismatched ends.

## Decision

Per-node provider routing via `app/services/llm/__init__.py::llm_service_for`:

| Node alias | Default model | Provider | Why |
|------------|---------------|----------|-----|
| `classifier` | `gemini-2.5-flash` | Gemini | Cheap + good structured output |
| `query_rewrite` | `gemini-2.5-flash` | Gemini | Same |
| `clarification` | `gemini-2.5-flash` | Gemini | Cheap chitchat |
| `answer_direct` | `gemini-2.5-flash` | Gemini | Cheap chitchat |
| `rag` | `claude-sonnet-4-5` (Bedrock) | Anthropic | Best grounding + citation faithfulness |
| `summarize` | `claude-haiku-4-5` (Bedrock) | Anthropic | Cheap Anthropic for long context summary |
| `eval_judge` | `gemini-2.5-flash` | Gemini | Cheap + JSON mode |

Provider is auto-detected from the model ID (Bedrock IDs prefix with
`us.anthropic.`, Gemini IDs start with `gemini-`).

All providers implement the same `LLMProvider` Protocol
(`generate`, `generate_structured`, `generate_stream`), so node code is
provider-agnostic.

## Why

- **Cost.** Gemini 2.5 Flash on cheap nodes saves ~10x on hot-path turns
  (classifier runs every turn).
- **Quality.** Anthropic Claude consistently outperforms on long-context
  document grounding and citation faithfulness in our eval suite. Cheaper
  providers hallucinate citations on dense chunks.
- **Resilience.** A Bedrock outage doesn't kill the classifier (Gemini), and
  vice versa. Degraded mode > full outage.
- **Override per-node via env.** `AGENT_MODEL_RAG_GENERATION=us.anthropic.claude-haiku-4-5-20251001-v1:0`
  swaps the RAG model without code change. Useful for cost experiments and
  for the eval suite's A/B mode.

## Trade-offs

- Two provider integrations to maintain. Mitigated by the
  `LLMProvider` Protocol — adding a third (e.g. Mistral, OpenAI direct)
  requires implementing only `generate / generate_structured / generate_stream`.
- Streaming protocols differ; `langchain-aws` and `langchain-google-genai`
  abstract the wire format but yield slightly different chunk shapes. Provider
  modules normalize to `LLMStreamDelta`.

## Cross-references

- `app/services/llm/base.py` defines the Protocol + `LLMResponse` /
  `LLMStreamDelta` shapes.
- `app/agent/utils/streaming.py::stream_generate` wraps any provider's
  `generate_stream` and dispatches `agent_token` LangGraph events for
  the SSE endpoint.
- `app/services/llm/pricing.py` carries the static price table — used for
  cost tracking on `query_logs.cost_usd` and in the eval runner.
