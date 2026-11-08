"""OpenTelemetry traces: one per agent turn (audit M-3, E11).

    invoke_workflow whisperdocs-agent      the turn                 INTERNAL
      check_compaction                     a graph node             INTERNAL
      decide_action
        chat gemini-3.8-flash              a model call             CLIENT
      search_docs
        embeddings text-embedding-3-small  an embedding request     CLIENT
        retrieval                          the retrieval pipeline   INTERNAL
        chat us.anthropic.claude-...

The turn span carries the sum of its calls' tokens and cost, the classifier's
and the summary's included (audit A5). Its trace id goes into the logs
(`app/logging_config.py`) and into `query_logs.trace_id`. `/query` and the MCP
server's `ask_documents` get an `invoke_workflow whisperdocs-query` span.

Attribute names follow the OpenTelemetry GenAI semantic conventions as of
open-telemetry/semantic-conventions-genai on 2026-11-06. They are in
Development and may still be renamed, so each name is spelled here, once.
No span records prompt, answer or document text: the conventions make that
opt-in, and the documents are the user's.

A TracerProvider is always installed, so every turn has a trace id. Spans
leave the process only when OTEL_EXPORTER_OTLP_ENDPOINT is set.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, SpanKind, StatusCode
from opentelemetry.util.re import parse_env_headers

from app.config import settings
from app.services.llm.model_policy import accepts_custom_sampling, reasoning_level
from app.services.llm.pricing import estimate_cost_usd

SERVICE_NAME = "whisperdocs"
AGENT_WORKFLOW = "whisperdocs-agent"
QUERY_WORKFLOW = "whisperdocs-query"

# GenAI semantic conventions (Development), 2026-11-06.
OPERATION = "gen_ai.operation.name"
PROVIDER = "gen_ai.provider.name"
AGENT_NAME = "gen_ai.agent.name"
CONVERSATION_ID = "gen_ai.conversation.id"
DATA_SOURCE_ID = "gen_ai.data_source.id"
OUTPUT_TYPE = "gen_ai.output.type"
REQUEST_MODEL = "gen_ai.request.model"
REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
REQUEST_TEMPERATURE = "gen_ai.request.temperature"
REQUEST_REASONING_LEVEL = "gen_ai.request.reasoning.level"
REQUEST_STREAM = "gen_ai.request.stream"
RESPONSE_MODEL = "gen_ai.response.model"
TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"
ERROR_TYPE = "error.type"
# General conventions. Langfuse groups a thread's traces into a session by it.
SESSION_ID = "session.id"
# Not in the conventions: Langfuse takes it as the call's cost, instead of
# pricing the model from its own table. Only model and embedding calls carry
# it, since Langfuse adds up a trace's costs itself.
CALL_COST = "gen_ai.usage.cost"
# This app's.
ROLE = "whisperdocs.role"
TURN_COST_USD = "whisperdocs.cost_usd"
CHANNEL = "whisperdocs.channel"
ACTION = "whisperdocs.action"
DOCUMENT_ID = "whisperdocs.document_id"
CHUNKS_RETURNED = "whisperdocs.retrieval.chunks"

tracer = trace.get_tracer(SERVICE_NAME)

_provider: TracerProvider | None = None


def build_tracer_provider() -> TracerProvider:
    """A TracerProvider that exports over OTLP/HTTP when an endpoint is set."""
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    endpoint = settings.otel_exporter_otlp_endpoint.strip()
    if endpoint:
        raw = settings.otel_exporter_otlp_headers
        headers = parse_env_headers(raw.get_secret_value(), liberal=True) if raw else None
        exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces", headers=headers)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def configure_tracing() -> TracerProvider:
    """Install this process's TracerProvider, once. It flushes at exit."""
    global _provider
    if _provider is None:
        _provider = build_tracer_provider()
        trace.set_tracer_provider(_provider)
    return _provider


def _trace_id(span: Span) -> str | None:
    context = span.get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None


def error_type(exc: BaseException) -> str:
    """`error.type`: the kind `classify_error` gives, low in cardinality."""
    from app.services.llm.errors import classify_error

    return classify_error(exc)


def _fail(span: Span, kind: str) -> None:
    span.set_attribute(ERROR_TYPE, kind)
    span.set_status(StatusCode.ERROR, kind)


def _present(attributes: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in attributes.items() if value is not None}


@dataclass
class _TurnTotals:
    conversation_id: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


# The turn in progress. Graph nodes run in tasks that copy the context, so
# they see the same totals object and add to it.
_turn: ContextVar[_TurnTotals | None] = ContextVar("whisperdocs_turn", default=None)


def _current_span(name: str, attributes: dict[str, Any]) -> AbstractContextManager[Span]:
    # Exceptions are recorded by the callers, as `error.type` without the
    # message: a provider's message can quote the prompt.
    return tracer.start_as_current_span(
        name,
        kind=SpanKind.INTERNAL,
        attributes=_present(attributes),
        record_exception=False,
        set_status_on_exception=False,
    )


@contextmanager
def _internal_span(name: str, attributes: dict[str, Any]) -> Iterator[Span]:
    with _current_span(name, attributes) as span:
        try:
            yield span
        except Exception as exc:
            _fail(span, error_type(exc))
            raise


class Workflow:
    """The span of one turn, as `workflow()` yields it."""

    def __init__(self, span: Span) -> None:
        self._span = span
        self.failed = False

    @property
    def trace_id(self) -> str | None:
        return _trace_id(self._span)

    def outcome(self, action: str | None, failure: str | None) -> None:
        """The action the turn took, and the kind of error when its answer
        failed without raising (`turn_errors.failure_kind`)."""
        if action:
            self._span.set_attribute(ACTION, action)
        if failure:
            self.fail(failure)

    def fail(self, kind: str) -> None:
        """A turn that failed without raising, or whose own kind should win
        over the one of the exception raised after it."""
        self.failed = True
        _fail(self._span, kind)


@contextmanager
def workflow(
    name: str,
    *,
    conversation_id: str | None = None,
    channel: str | None = None,
    collection_id: int | None = None,
    document_id: int | None = None,
) -> Iterator[Workflow]:
    """The span of a turn: `invoke_workflow {name}`. The graph decides the
    control flow, not the model, so the conventions call it a workflow.

    It sums the tokens and cost of the calls made inside it, and puts
    `thread_id` into the log context while it lasts.
    """
    totals = _TurnTotals(conversation_id)
    token = _turn.set(totals)
    log_context = {"thread_id": conversation_id} if conversation_id else {}
    try:
        with (
            _current_span(
                f"invoke_workflow {name}",
                {
                    OPERATION: "invoke_workflow",
                    AGENT_NAME: name,
                    CONVERSATION_ID: conversation_id,
                    SESSION_ID: conversation_id,
                    CHANNEL: channel,
                    DATA_SOURCE_ID: str(collection_id) if collection_id is not None else None,
                    DOCUMENT_ID: document_id,
                },
            ) as span,
            structlog.contextvars.bound_contextvars(**log_context),
        ):
            turn = Workflow(span)
            try:
                yield turn
            except Exception as exc:
                if not turn.failed:
                    turn.fail(error_type(exc))
                raise
            finally:
                span.set_attributes(
                    {
                        INPUT_TOKENS: totals.input_tokens,
                        OUTPUT_TOKENS: totals.output_tokens,
                        TURN_COST_USD: round(totals.cost_usd, 6),
                    }
                )
    finally:
        try:
            _turn.reset(token)
        except ValueError:
            # A stream closed from another context: nothing left to restore.
            pass


def traced_node(name: str, node: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
    """`node` inside a span named after it. `functools.wraps` keeps its
    signature, which LangGraph reads to decide whether to pass `config`."""

    @functools.wraps(node)
    async def traced(*args: Any, **kwargs: Any) -> dict:
        with _internal_span(name, {}):
            return await node(*args, **kwargs)

    return traced


class Retrieval:
    def __init__(self, span: Span) -> None:
        self._span = span

    def returned(self, chunks: int) -> None:
        self._span.set_attribute(CHUNKS_RETURNED, chunks)


@contextmanager
def retrieval(
    *, top_k: int, collection_id: int | None, document_id: int | None
) -> Iterator[Retrieval]:
    with _internal_span(
        "retrieval",
        {
            OPERATION: "retrieval",
            RETRIEVAL_TOP_K: top_k,
            DATA_SOURCE_ID: str(collection_id) if collection_id is not None else None,
            DOCUMENT_ID: document_id,
        },
    ) as span:
        yield Retrieval(span)


class ModelCall:
    """A model or embedding call's span, as `chat()` and `embeddings()` yield it."""

    def __init__(self, span: Span, model: str) -> None:
        self._span = span
        self._model = model
        self._started = time.perf_counter()
        self._first_chunk = False

    def first_chunk(self) -> None:
        if not self._first_chunk:
            self._first_chunk = True
            self._span.set_attribute(TIME_TO_FIRST_CHUNK, time.perf_counter() - self._started)

    def usage(
        self, input_tokens: int, output_tokens: int | None = None, model: str | None = None
    ) -> None:
        """The tokens the call used, priced with `pricing.py` and added to the
        turn. An embedding request has input tokens only."""
        cost = estimate_cost_usd(model or self._model, input_tokens, output_tokens or 0)
        self._span.set_attributes(
            _present(
                {
                    RESPONSE_MODEL: model,
                    INPUT_TOKENS: input_tokens,
                    OUTPUT_TOKENS: output_tokens,
                    CALL_COST: cost,
                }
            )
        )
        turn = _turn.get()
        if turn is not None:
            turn.input_tokens += input_tokens
            turn.output_tokens += output_tokens or 0
            turn.cost_usd += cost


@contextmanager
def _client_call(
    operation: str, *, provider: str, model: str, attributes: dict[str, Any]
) -> Iterator[ModelCall]:
    """A CLIENT span that is not made current: nothing runs inside a model
    call, and a streaming call's span stays open across the reader's yields."""
    turn = _turn.get()
    span = tracer.start_span(
        f"{operation} {model}",
        kind=SpanKind.CLIENT,
        attributes=_present(
            {
                OPERATION: operation,
                PROVIDER: provider,
                REQUEST_MODEL: model,
                CONVERSATION_ID: turn.conversation_id if turn else None,
                **attributes,
            }
        ),
    )
    try:
        yield ModelCall(span, model)
    except Exception as exc:
        _fail(span, error_type(exc))
        raise
    finally:
        span.end()


def chat(
    *,
    provider: str,
    model: str,
    role: str,
    temperature: float,
    max_tokens: int,
    stream: bool = False,
    structured: bool = False,
) -> AbstractContextManager[ModelCall]:
    """The span of a model call. The temperature and the reasoning level are
    recorded only when the provider sends them (`model_policy.py`)."""
    return _client_call(
        "chat",
        provider=provider,
        model=model,
        attributes={
            ROLE: role,
            REQUEST_MAX_TOKENS: max_tokens,
            REQUEST_TEMPERATURE: temperature if accepts_custom_sampling(model) else None,
            REQUEST_REASONING_LEVEL: reasoning_level(model),
            REQUEST_STREAM: stream,
            OUTPUT_TYPE: "json" if structured else None,
        },
    )


def embeddings(*, provider: str, model: str) -> AbstractContextManager[ModelCall]:
    """The span of one embedding request (a batch of up to 100 texts)."""
    return _client_call("embeddings", provider=provider, model=model, attributes={})
