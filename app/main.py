import asyncio
import gc
import sys
from contextlib import asynccontextmanager

# asyncpg on Windows with ProactorEventLoop resets during Postgres TLS
# handshake. Force Selector policy before any loop is created.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.checkpointer import close_checkpointer, init_checkpointer
from app.agent.graph import build_agent_graph
from app.config import settings
from app.logging_config import configure_logging, get_logger
from app.services.llm import validate_agent_models
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
)
from app.api.v1.webhooks import whatsapp as webhook_whatsapp


configure_logging()
logger = get_logger(__name__)


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
    checkpointer = await init_checkpointer()
    app.state.checkpointer = checkpointer
    app.state.agent = build_agent_graph().compile(checkpointer=checkpointer)
    # Keep what startup allocated (modules, the compiled graph, about 300k
    # objects) out of the collector. A full collection holds the GIL, so it
    # pauses the event loop while it scans. A long PDF parse triggers several,
    # and freezing halved the longest pause (audit R4).
    gc.collect()
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()
        await close_checkpointer()


app = FastAPI(
    title="WhisperDocs",
    description="Agentic RAG system with WhatsApp integration",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(metrics_route.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(collections.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(eval.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")
app.include_router(conversations.router, prefix="/api/v1")
app.include_router(notes.router, prefix="/api/v1")
app.include_router(webhook_whatsapp.router, prefix="/api/v1")
