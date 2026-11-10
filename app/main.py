import asyncio
import gc
import sys
from contextlib import asynccontextmanager

# asyncpg on Windows with ProactorEventLoop resets during Postgres TLS
# handshake. Force Selector policy before any loop is created.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.checkpointer import (
    close_checkpointer,
    init_checkpointer,
    start_pruning,
    stop_pruning,
)
from app.api.auth import configured_key, require_api_key
from app.api.body_limit import BodySizeLimitMiddleware
from app.api.errors import RequestContextMiddleware, install_error_handlers
from app.api.http_metrics import HTTPMetricsMiddleware
from app.agent.graph import build_agent_graph
from app.config import settings
from app.logging_config import configure_logging, get_logger
from app.mcp.http import MCP_PATH, McpOverHttp
from app.services.ingestion import fail_interrupted_jobs, stop_ingestions
from app.services.llm import validate_agent_models
from app.services.whatsapp.inbound import stop_replies
from app.telemetry import configure_tracing
from app.api.v1 import (
    admin,
    agent,
    collections,
    conversations,
    documents,
    eval,
    health,
    metrics as metrics_route,
    notes,
    query,
    retrieve,
)
from app.api.v1.webhooks import whatsapp as webhook_whatsapp


configure_logging()
configure_tracing()
logger = get_logger(__name__)

# The MCP server over Streamable HTTP (audit M-1, M-2): see app/mcp/http.py.
mcp_http = McpOverHttp()


def _check_agent_models() -> None:
    """Fail at startup on a model configuration that fails on every call."""
    errors, warnings = validate_agent_models()
    for message in warnings:
        logger.warning("agent_model_config_warning", detail=message)
    if errors:
        for message in errors:
            logger.error("agent_model_config_error", detail=message)
        raise RuntimeError("Invalid agent model configuration: " + "; ".join(errors))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_agent_models()
    if configured_key() is None:
        logger.warning(
            "api_key_not_set",
            detail="every route is open; set API_KEY before the API is reachable from elsewhere",
        )
    interrupted = await fail_interrupted_jobs()
    if interrupted:
        logger.warning("ingestion_jobs_interrupted", count=interrupted)
    checkpointer = await init_checkpointer()
    app.state.checkpointer = checkpointer
    app.state.agent = build_agent_graph().compile(checkpointer=checkpointer)
    start_pruning(checkpointer)
    # Keep what startup allocated (modules, the compiled graph, about 300k
    # objects) out of the collector. A full collection holds the GIL, so it
    # pauses the event loop while it scans. A long PDF parse triggers several,
    # and freezing halved the longest pause (audit R4).
    gc.collect()
    gc.freeze()
    try:
        async with mcp_http.running():
            yield
    finally:
        gc.unfreeze()
        # Together: each waits up to 5 s, and Docker kills the process at 10.
        cancelled, unanswered = await asyncio.gather(stop_ingestions(), stop_replies())
        if cancelled:
            logger.warning("ingestion_jobs_cancelled_on_shutdown", count=cancelled)
        if unanswered:
            logger.warning("whatsapp_replies_cancelled_on_shutdown", count=unanswered)
        await stop_pruning()
        await close_checkpointer()


app = FastAPI(
    title="WhisperDocs",
    description="Agentic RAG system with WhatsApp integration",
    version="0.1.0",
    lifespan=lifespan,
    # FastAPI's own OpenTelemetry: a span per HTTP request, the root of a
    # turn's trace, on the provider `configure_tracing` installed. Its
    # exporters stay off (app/telemetry.py has the one), as do its dependency,
    # endpoint and serialization spans, its metrics (Prometheus has them) and
    # its logs, which would carry exception messages.
    telemetry={
        "auto_configure": False,
        "operation_spans": False,
        "metrics": False,
        "logs": False,
    },
)

API_PREFIX = "/api/v1"

install_error_handlers(app)

# Each add_middleware wraps the ones before it. CORS goes last, outermost, so
# that a 413 from the body limit or a 500 from the request context still
# carries the CORS headers, and the browser shows them instead of a CORS
# error. The request context wraps the body limit, so its 413 has a request id.
app.add_middleware(
    BodySizeLimitMiddleware,
    limits={API_PREFIX + path: limit for path, limit in documents.BODY_LIMITS.items()},
)
app.add_middleware(RequestContextMiddleware)
# Outside the request context, so an unhandled exception counts as the 500 it
# answers; inside CORS, so preflights are not counted.
app.add_middleware(HTTPMetricsMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Readable by the page: the id an error quotes, and a 429's wait.
    expose_headers=["X-Request-ID", "Retry-After"],
)

# Without the key: /health checks it itself, for `deep` only, and the
# WhatsApp webhook checks Twilio's signature.
app.include_router(health.router, prefix=API_PREFIX)
app.include_router(webhook_whatsapp.router, prefix=API_PREFIX)

for router in (
    metrics_route.router,
    admin.router,
    collections.router,
    documents.router,
    query.router,
    retrieve.router,
    eval.router,
    agent.router,
    conversations.router,
    notes.router,
):
    app.include_router(router, prefix=API_PREFIX, dependencies=[Depends(require_api_key)])

# Its own key check (app/mcp/access.py): an MCP key reaches some collections.
app.add_route(MCP_PATH, mcp_http, include_in_schema=False)
