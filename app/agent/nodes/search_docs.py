from langchain_core.messages import AIMessage, HumanMessage

from app.agent.prompts import RAG_SYSTEM_PROMPT_AGENT
from app.agent.schemas import RAGAnswer
from app.agent.state import AgentState
from app.agent.utils.citation_validation import validate_citations_semantic
from app.agent.utils.messages import build_prompt_messages
from app.agent.utils.query_rewrite import rewrite_query_for_retrieval
from app.agent.utils.streaming import stream_generate
from app.config import settings
from app.db import SessionLocal
from app.logging_config import get_logger
from app.metrics import (
    errors_total,
    llm_tokens_total,
    retrieval_chunks,
    retrieval_top_relevance,
)
from app.services.embedder import EmbeddingService
from app.services.llm import llm_service_for
from app.services.rag import build_context_block, validate_citations
from app.services.retrieval_pipeline import retrieve as retrieve_pipeline
from app.services.retriever import RetrievedChunk

logger = get_logger(__name__)


async def search_docs_node(state: AgentState) -> dict:
    """Query rewrite → vector retrieval → grounded RAG generation with citations.

    Single LLM tool call: rewrites the user query for retrieval, fetches top-k
    chunks via pgvector, builds a grounded prompt, generates the answer with
    inline citations. Opens its own DB session — agent runs outside FastAPI request cycle.
    """
    messages = state["messages"]
    covered = state.get("summary_covered_count") or 0
    uncovered = messages[covered:]
    last_user_msg = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
        "",
    )
    collection_id = state.get("collection_id")
    state_summary = state.get("summary")

    # 1. Query rewrite
    retrieval_query = await rewrite_query_for_retrieval(
        history=uncovered,
        current=str(last_user_msg),
        summary=state_summary,
    )

    # 2. Embedding + retrieval
    embedder = EmbeddingService()
    try:
        query_embedding = await embedder.embed(retrieval_query)
    except Exception:
        logger.exception("embedding_failed", node="search_docs")
        query_embedding = None

    chunks: list[RetrievedChunk] = []
    if query_embedding is not None:
        async with SessionLocal() as db:
            try:
                chunks = await retrieve_pipeline(
                    db=db,
                    query_text=retrieval_query,
                    query_embedding=query_embedding,
                    collection_id=collection_id,
                    top_k=settings.agent.retrieval_top_k,
                )
            except Exception:
                errors_total.labels(type="retrieval").inc()
                logger.exception(
                    "retrieval_failed",
                    node="search_docs",
                    collection_id=collection_id,
                    top_k=settings.agent.retrieval_top_k,
                )
                chunks = []

    top_rel = chunks[0].relevance if chunks else 0.0
    retrieval_chunks.observe(len(chunks))
    retrieval_top_relevance.observe(top_rel)

    logger.info(
        "retrieval_done",
        chunks=len(chunks),
        top_relevance=top_rel,
        collection_id=collection_id,
    )

    # 3. Generation
    if chunks:
        context_block, included_chunks = build_context_block(
            chunks, token_budget=settings.agent.rag_context_token_budget
        )
    else:
        context_block, included_chunks = "(no context available)", []
    rag_user_prompt = (
        f"Context:\n\n{context_block}\n\n---\n\n"
        f"Question: {last_user_msg}\n\nAnswer:"
    )

    provider, model, temperature = llm_service_for("rag")
    history_for_llm = build_prompt_messages(state, exclude_last=True)
    history_for_llm.append({"role": "user", "content": rag_user_prompt})

    structured_md: dict = {}

    try:
        if settings.agent.rag_structured_output:
            parsed, resp = await provider.generate_structured(
                system=RAG_SYSTEM_PROMPT_AGENT,
                messages=history_for_llm,
                model=model,
                response_model=RAGAnswer,
                temperature=temperature,
                max_tokens=1000,
            )
            answer = (parsed.answer or "").strip() or (
                "No tengo información suficiente para responder esto."
            )
            structured_md = {
                "rag_uses_context": parsed.uses_context,
                "rag_confidence": parsed.confidence,
                "rag_cited_chunks": parsed.cited_chunks,
                "rag_refused": parsed.refused,
                "rag_refuse_reason": parsed.refuse_reason,
            }
            logger.info(
                "rag_structured_generated",
                uses_context=parsed.uses_context,
                confidence=parsed.confidence,
                cited=parsed.cited_chunks,
                refused=parsed.refused,
            )
        else:
            resp = await stream_generate(
                provider,
                system=RAG_SYSTEM_PROMPT_AGENT,
                messages=history_for_llm,
                model=model,
                temperature=temperature,
                max_tokens=1000,
                node="search_docs",
            )
            answer = resp.text.strip() or "No tengo información suficiente para responder esto."
        answer, _used = validate_citations(answer, len(included_chunks))
        llm_tokens_total.labels(model=resp.model, direction="input").inc(resp.tokens_input)
        llm_tokens_total.labels(model=resp.model, direction="output").inc(resp.tokens_output)
        gen_md = {
            "generation_model": resp.model,
            "generation_tokens_input": resp.tokens_input,
            "generation_tokens_output": resp.tokens_output,
            "generation_latency_ms": resp.latency_ms,
        }
    except Exception:
        errors_total.labels(type="rag_generation").inc()
        logger.exception("rag_generation_failed", model=model, chunks=len(chunks))
        answer = "Tuve un problema generando la respuesta. Probá de nuevo en un momento."
        gen_md = {"generation_model": model}

    if (
        settings.agent.citation_semantic_validation
        and included_chunks
        and answer
    ):
        try:
            validation = await validate_citations_semantic(
                answer=answer,
                included_chunks=included_chunks,
                embedder=embedder,
                min_similarity=settings.agent.citation_semantic_min_similarity,
            )
            low_conf = validation["low_confidence_refs"]
            if low_conf:
                structured_md["citation_low_confidence_refs"] = low_conf
                structured_md["citation_min_similarity"] = (
                    settings.agent.citation_semantic_min_similarity
                )
        except Exception:
            logger.exception("citation_semantic_validation_failed")

    citations = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "document_title": c.document_title,
            "chunk_index": c.chunk_index,
            "relevance": c.relevance,
            "text_preview": (c.text[:500] + "...") if len(c.text) > 500 else c.text,
        }
        for c in included_chunks
    ]
    msg_metadata = {
        "model": gen_md.get("generation_model"),
        "tokens_input": gen_md.get("generation_tokens_input", 0),
        "tokens_output": gen_md.get("generation_tokens_output", 0),
        "latency_ms": gen_md.get("generation_latency_ms", 0),
        "chunks_retrieved": len(chunks),
    }
    prev = state.get("metadata") or {}
    return {
        "messages": [
            AIMessage(
                content=answer,
                additional_kwargs={"metadata": msg_metadata, "citations": citations},
            )
        ],
        "retrieved_chunks": citations,
        "metadata": {
            **prev,
            **gen_md,
            **structured_md,
            "collection_id_active": collection_id,
            "retrieval_chunks_count": len(chunks),
            "retrieval_top_relevance": chunks[0].relevance if chunks else 0.0,
        },
    }
