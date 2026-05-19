import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionLocal, get_db
from app.logging_config import get_logger
from app.models import Collection, Conversation, QueryLog
from app.services.llm.pricing import estimate_cost_usd
from app.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse,
    AgentCitation,
    AgentTurnMetadata,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(
    payload: AgentChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AgentChatResponse:
    if payload.collection_id is not None:
        exists = await db.scalar(
            select(Collection.id).where(Collection.id == payload.collection_id)
        )
        if exists is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Collection {payload.collection_id} not found",
            )

    thread_id = str(payload.thread_id) if payload.thread_id else str(uuid.uuid4())

    conv = await db.scalar(select(Conversation).where(Conversation.thread_id == thread_id))
    if conv is None:
        conv = Conversation(
            thread_id=thread_id,
            collection_id=payload.collection_id,
            channel="web",
        )
        db.add(conv)
        await db.flush()
    else:
        # Request can override collection per turn (§2.4)
        if payload.collection_id is not None:
            conv.collection_id = payload.collection_id

    agent = request.app.state.agent
    try:
        result = await agent.ainvoke(
            {
                "messages": [HumanMessage(content=payload.message)],
                "collection_id": conv.collection_id,
                "top_k": payload.top_k,
            },
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 10,
            },
        )
    except Exception as exc:
        logger.exception("agent_invoke_failed", thread_id=thread_id)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Agent failure: {exc}")

    final = result["messages"][-1]
    answer = final.content if isinstance(final.content, str) else str(final.content)
    action = result.get("pending_action")
    retrieved = result.get("retrieved_chunks") or []
    md = result.get("metadata") or {}

    conv.last_active_at = func.now()

    if action == "search_docs":
        gen_model = md.get("generation_model") or "agent"
        tin = md.get("generation_tokens_input", 0)
        tout = md.get("generation_tokens_output", 0)
        db.add(
            QueryLog(
                question=payload.message,
                answer=answer,
                collection_id=conv.collection_id,
                chunks_retrieved=md.get("retrieval_chunks_count", len(retrieved)),
                model_used=gen_model,
                tokens_input=tin,
                tokens_output=tout,
                latency_ms=md.get("generation_latency_ms", 0),
                cost_usd=estimate_cost_usd(gen_model, tin, tout),
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
        metadata=AgentTurnMetadata(
            model=md.get("generation_model"),
            tokens_input=md.get("generation_tokens_input", 0),
            tokens_output=md.get("generation_tokens_output", 0),
            latency_ms=md.get("generation_latency_ms", 0),
            chunks_retrieved=md.get("retrieval_chunks_count", len(retrieved)),
        ),
    )


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


@router.post("/chat/stream")
async def agent_chat_stream(
    payload: AgentChatRequest,
    request: Request,
) -> StreamingResponse:
    """Server-sent events stream of agent execution.

    Events:
      - thread: {thread_id}                      (once, at start)
      - node_start: {node}                       (each node entry)
      - node_end: {node, summary?}               (each node exit, with light summary)
      - final: {action, answer, citations, metadata}
      - error: {message}
    """

    async def event_stream() -> AsyncIterator[bytes]:
        thread_id = str(payload.thread_id) if payload.thread_id else str(uuid.uuid4())
        collection_id = payload.collection_id

        async with SessionLocal() as db:
            if collection_id is not None:
                exists = await db.scalar(
                    select(Collection.id).where(Collection.id == collection_id)
                )
                if exists is None:
                    yield _sse(
                        "error", {"message": f"Collection {collection_id} not found"}
                    )
                    return

            conv = await db.scalar(
                select(Conversation).where(Conversation.thread_id == thread_id)
            )
            if conv is None:
                conv = Conversation(
                    thread_id=thread_id,
                    collection_id=collection_id,
                    channel="web",
                )
                db.add(conv)
                await db.flush()
            elif collection_id is not None:
                conv.collection_id = collection_id
            await db.commit()
            active_collection_id = conv.collection_id

        yield _sse("thread", {"thread_id": thread_id})

        agent = request.app.state.agent

        final_state: dict = {}
        try:
            async for event in agent.astream_events(
                {
                    "messages": [HumanMessage(content=payload.message)],
                    "collection_id": active_collection_id,
                    "top_k": payload.top_k,
                },
                config={
                    "configurable": {"thread_id": thread_id},
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
            yield _sse("error", {"message": f"Agent failure: {exc}"})
            return

        if not final_state:
            yield _sse("error", {"message": "Agent returned no final state"})
            return

        messages = final_state.get("messages") or []
        final_msg = messages[-1] if messages else None
        answer = ""
        if final_msg is not None:
            content = getattr(final_msg, "content", "")
            answer = content if isinstance(content, str) else str(content)

        action = final_state.get("pending_action")
        retrieved = final_state.get("retrieved_chunks") or []
        md = final_state.get("metadata") or {}
        chunks_count = md.get("retrieval_chunks_count", len(retrieved))

        async with SessionLocal() as db:
            conv = await db.scalar(
                select(Conversation).where(Conversation.thread_id == thread_id)
            )
            if conv is not None:
                conv.last_active_at = func.now()
                if action == "search_docs":
                    gen_model = md.get("generation_model") or "agent"
                    tin = md.get("generation_tokens_input", 0)
                    tout = md.get("generation_tokens_output", 0)
                    db.add(
                        QueryLog(
                            question=payload.message,
                            answer=answer,
                            collection_id=conv.collection_id,
                            chunks_retrieved=chunks_count,
                            model_used=gen_model,
                            tokens_input=tin,
                            tokens_output=tout,
                            latency_ms=md.get("generation_latency_ms", 0),
                            cost_usd=estimate_cost_usd(gen_model, tin, tout),
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
                "metadata": {
                    "model": md.get("generation_model"),
                    "tokens_input": md.get("generation_tokens_input", 0),
                    "tokens_output": md.get("generation_tokens_output", 0),
                    "latency_ms": md.get("generation_latency_ms", 0),
                    "chunks_retrieved": chunks_count,
                },
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
