import asyncio
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
from app.logging_config import configure_logging
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer = await init_checkpointer()
    app.state.checkpointer = checkpointer
    app.state.agent = build_agent_graph().compile(checkpointer=checkpointer)
    try:
        yield
    finally:
        await close_checkpointer()


app = FastAPI(
    title="WhisperDocs",
    description="Agentic RAG system with WhatsApp integration",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
