import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.thread_id import thread_title
from app.agent.utils.messages import stamped
from app.api.errors import APIError, request_id
from app.api.rate_limit import questions_limit
from app.db import SessionLocal, get_db
from app.logging_config import get_logger
from app.metrics import agent_turn_duration_seconds
from app.models import Collection, Conversation, Document, QueryLog
from app.services.llm.errors import classify_error, failure_text
from app.services.llm.pricing import estimate_cost_usd
from app.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse,
    AgentCitation,
    AgentTurnMetadata,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


def _turn_metadata(md: dict, chunks_retrieved: int) -> AgentTurnMetadata:
    """The answer's model call, as the UI shows it and `query_logs` stores it."""
    model = md.get("generation_model")
    tokens_input = md.get("generation_tokens_input", 0)
    tokens_output = md.get("generation_tokens_output", 0)
    return AgentTurnMetadata(
        model=model,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        latency_ms=md.get("generation_latency_ms", 0),
        chunks_retrieved=chunks_retrieved,
        cost_usd=estimate_cost_usd(model, tokens_input, tokens_output),
    )


def _agent_failure(exc: BaseException) -> tuple[str, str]:
    """The message and kind of an agent run that raised. The exception's own
    text (Bedrock's, the database's) stays in the logs (audit A9)."""
    return f"The agent failed: {failure_text(exc)}.", classify_error(exc)


@router.post("/chat", response_model=AgentChatResponse, dependencies=[Depends(questions_limit)])
async def agent_chat(
    payload: AgentChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AgentChatResponse:
    document, error = await check_turn_scope(db, payload)
    if error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)

    thread_id = str(payload.thread_id) if payload.thread_id else str(uuid.uuid4())

    conv = await db.scalar(select(Conversation).where(Conversation.thread_id == thread_id))
    if conv is None:
        conv = Conversation(thread_id=thread_id, channel="web")
        apply_turn_scope(conv, payload, document)
        db.add(conv)
        await db.flush()
    else:
        apply_turn_scope(conv, payload, document)
    ensure_title(conv, payload.message)

    agent = request.app.state.agent
    turn_started = time.perf_counter()
    try:
        result = await agent.ainvoke(
            {
                "messages": [HumanMessage(content=payload.message, additional_kwargs=stamped())],
                "collection_id": conv.collection_id,
                "document_id": conv.document_id,
            },
            config={
                # top_k is per request, not conversation state (audit A6).
                "configurable": {"thread_id": thread_id, "top_k": payload.top_k},
                "recursion_limit": 10,
            },
        )
    except Exception as exc:
        logger.exception("agent_invoke_failed", thread_id=thread_id)
        message, kind = _agent_failure(exc)
        raise APIError(status.HTTP_502_BAD_GATEWAY, "agent_failed", message, kind=kind) from exc

    final = result["messages"][-1]
    answer = final.content if isinstance(final.content, str) else str(final.content)
    action = result.get("pending_action")
    agent_turn_duration_seconds.labels(action=action or "unknown").observe(
        time.perf_counter() - turn_started
    )
    retrieved = result.get("retrieved_chunks") or []
    md = result.get("metadata") or {}
    turn = _turn_metadata(md, md.get("retrieval_chunks_count", len(retrieved)))

    conv.last_active_at = func.now()

    # A failed turn has no answer to log: the failure is in errors_total and
    # the logs (audit N4).
    if action == "search_docs" and not md.get("turn_failed"):
        db.add(
            QueryLog(
                question=payload.message,
                answer=answer,
                collection_id=conv.collection_id,
                chunks_retrieved=turn.chunks_retrieved,
                model_used=turn.model or "agent",
                tokens_input=turn.tokens_input,
                tokens_output=turn.tokens_output,
                latency_ms=turn.latency_ms,
                cost_usd=turn.cost_usd,
                session_id=thread_id,
                agent_action=action,
            )
        )

    await db.commit()

    return AgentChatResponse(
        thread_id=thread_id,
        action=action,
        answer=answer,
        citations=[AgentCitation(**c) for c in retrieved],
        metadata=turn,
        failed=bool(md.get("turn_failed")),
        errors=md.get("turn_errors") or [],
    )


async def check_turn_scope(
    db: AsyncSession, payload: AgentChatRequest
) -> tuple[Document | None, str | None]:
    """The document the turn names, if any, and the error for a collection or
    a document that does not exist."""
    if payload.collection_id is not None:
        exists = await db.scalar(
            select(Collection.id).where(Collection.id == payload.collection_id)
        )
        if exists is None:
            return None, f"Collection {payload.collection_id} not found"
    if payload.document_id is None:
        return None, None
    document = await db.get(Document, payload.document_id)
    if document is None:
        return None, f"Document {payload.document_id} not found"
    return document, None


def apply_turn_scope(
    conv: Conversation, payload: AgentChatRequest, document: Document | None
) -> None:
    """A turn that names `collection_id` moves the thread to it, and an
    explicit null moves it to every collection (ALL); a turn that leaves the
    field out keeps the thread's collection (§2.4).

    `document_id` works the same way, one level down. A document narrows the
    thread to it, and moves the thread to the document's collection, so a
    later null goes back to that collection; null searches the collection
    again. While the thread has a document, retrieval searches only it.
    """
    if "collection_id" in payload.model_fields_set:
        conv.collection_id = payload.collection_id
    if "document_id" in payload.model_fields_set:
        conv.document_id = payload.document_id
        if document is not None:
            conv.collection_id = document.collection_id


def ensure_title(conv: Conversation, message: str) -> None:
    """A thread without a title takes this message: its first question, or
    for a thread from before titles, the first one since."""
    if conv.title is None:
        conv.title = thread_title(message)


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


@router.post("/chat/stream", dependencies=[Depends(questions_limit)])
async def agent_chat_stream(
    payload: AgentChatRequest,
    request: Request,
) -> StreamingResponse:
    """Server-sent events stream of agent execution.

    Events:
      - thread: {thread_id}                      (once, at start)
      - node_start: {node}                       (each node entry)
      - node_end: {node, summary?}               (each node exit, with light summary)
      - final: {thread_id, action, answer, citations, metadata, failed, errors}
      - error: {message, code, request_id, kind?}: the same fields as an
        error response (app/api/errors.py), without the exception's text.
    """
    rid = request_id(request.scope)

    def error_event(message: str, code: str, **extra: str) -> bytes:
        if code == "agent_failed":
            message = f"{message} Request id {rid}."
        return _sse("error", {"message": message, "code": code, "request_id": rid, **extra})

    async def event_stream() -> AsyncIterator[bytes]:
        thread_id = str(payload.thread_id) if payload.thread_id else str(uuid.uuid4())

        async with SessionLocal() as db:
            document, error = await check_turn_scope(db, payload)
            if error:
                yield error_event(error, "bad_request")
                return

            conv = await db.scalar(select(Conversation).where(Conversation.thread_id == thread_id))
            if conv is None:
                conv = Conversation(thread_id=thread_id, channel="web")
                apply_turn_scope(conv, payload, document)
                db.add(conv)
                await db.flush()
            else:
                apply_turn_scope(conv, payload, document)
            ensure_title(conv, payload.message)
            await db.commit()
            active_collection_id = conv.collection_id
            active_document_id = conv.document_id

        yield _sse("thread", {"thread_id": thread_id})

        agent = request.app.state.agent

        final_state: dict = {}
        turn_started = time.perf_counter()
        try:
            async for event in agent.astream_events(
                {
                    "messages": [
                        HumanMessage(content=payload.message, additional_kwargs=stamped())
                    ],
                    "collection_id": active_collection_id,
                    "document_id": active_document_id,
                },
                config={
                    "configurable": {"thread_id": thread_id, "top_k": payload.top_k},
                    "recursion_limit": 10,
                },
                version="v2",
            ):
                kind = event["event"]
                name = event.get("name", "")
                if kind == "on_custom_event" and name == "agent_token":
                    payload_data = event.get("data") or {}
                    yield _sse(
                        "token",
                        {
                            "node": payload_data.get("node"),
                            "delta": payload_data.get("delta", ""),
                        },
                    )
                    continue
                if kind == "on_chain_start" and name in {
                    "check_compaction",
                    "summarize",
                    "decide_action",
                    "search_docs",
                    "ask_clarification",
                    "answer_direct",
                }:
                    yield _sse("node_start", {"node": name})
                elif kind == "on_chain_end" and name in {
                    "check_compaction",
                    "summarize",
                    "decide_action",
                    "search_docs",
                    "ask_clarification",
                    "answer_direct",
                }:
                    output = event.get("data", {}).get("output") or {}
                    summary: dict = {}
                    if name == "decide_action":
                        summary["action"] = output.get("pending_action")
                    if name == "search_docs":
                        meta = output.get("metadata") or {}
                        summary["chunks"] = meta.get("retrieval_chunks_count", 0)
                    yield _sse("node_end", {"node": name, "summary": summary})
                elif kind == "on_chain_end" and name == "LangGraph":
                    final_state = event.get("data", {}).get("output") or {}
        except Exception as exc:
            logger.exception("agent_stream_failed", thread_id=thread_id)
            message, kind = _agent_failure(exc)
            yield error_event(message, "agent_failed", kind=kind)
            return

        if not final_state:
            yield error_event("The agent returned no answer.", "agent_failed")
            return

        messages = final_state.get("messages") or []
        final_msg = messages[-1] if messages else None
        answer = ""
        if final_msg is not None:
            content = getattr(final_msg, "content", "")
            answer = content if isinstance(content, str) else str(content)

        action = final_state.get("pending_action")
        agent_turn_duration_seconds.labels(action=action or "unknown").observe(
            time.perf_counter() - turn_started
        )
        retrieved = final_state.get("retrieved_chunks") or []
        md = final_state.get("metadata") or {}
        turn = _turn_metadata(md, md.get("retrieval_chunks_count", len(retrieved)))

        async with SessionLocal() as db:
            conv = await db.scalar(select(Conversation).where(Conversation.thread_id == thread_id))
            if conv is not None:
                conv.last_active_at = func.now()
                if action == "search_docs" and not md.get("turn_failed"):
                    db.add(
                        QueryLog(
                            question=payload.message,
                            answer=answer,
                            collection_id=conv.collection_id,
                            chunks_retrieved=turn.chunks_retrieved,
                            model_used=turn.model or "agent",
                            tokens_input=turn.tokens_input,
                            tokens_output=turn.tokens_output,
                            latency_ms=turn.latency_ms,
                            cost_usd=turn.cost_usd,
                            session_id=thread_id,
                            agent_action=action,
                        )
                    )
                await db.commit()

        # Mark the assistant message as the source of truth for assertion in tests
        if isinstance(final_msg, AIMessage):
            pass

        yield _sse(
            "final",
            {
                "thread_id": thread_id,
                "action": action,
                "answer": answer,
                "citations": list(retrieved),
                "metadata": turn.model_dump(),
                "failed": bool(md.get("turn_failed")),
                "errors": list(md.get("turn_errors") or []),
            },
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
