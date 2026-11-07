from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.agent.prompts import RAG_SYSTEM_PROMPT_AGENT
from app.agent.schemas import RAGAnswer
from app.agent.state import AgentState
from app.agent.utils.citation_validation import validate_citations_semantic
from app.agent.utils.fixed_replies import FixedReply, fixed_reply
from app.agent.utils.messages import build_prompt_messages, failed_answer
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
from app.services.rag import build_user_prompt, validate_citations
from app.services.retrieval_pipeline import retrieve as retrieve_pipeline
from app.services.retriever import RetrievedChunk

logger = get_logger(__name__)

# Turns answered without a model call (audit A12) get a fixed reply. The
# vector leg returns the nearest chunks whatever their similarity, so an empty
# result means nothing is indexed in scope, not that nothing was relevant.
_NO_GENERATION = {
    "generation_model": None,
    "generation_tokens_input": 0,
    "generation_tokens_output": 0,
    "generation_latency_ms": 0,
}


def _requested_top_k(config: RunnableConfig | None) -> int:
    """`top_k` the client sent, in `config["configurable"]` (audit A6).

    It is a per-request value, not conversation state, so it travels in the
    run config and not in AgentState. WhatsApp and direct calls send none and
    get `AGENT_RETRIEVAL_TOP_K`.
    """
    configurable = (config or {}).get("configurable") or {}
    return configurable.get("top_k") or settings.agent.retrieval_top_k


async def search_docs_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Query rewrite → vector retrieval → grounded RAG generation with citations.

    Single LLM tool call: rewrites the user query for retrieval, fetches top-k
    chunks via pgvector, builds a grounded prompt, generates the answer with
    inline citations. Opens its own DB session — agent runs outside FastAPI request cycle.
    Without chunks, or when retrieval fails, it answers without calling the model.
    """
    messages = state["messages"]
    covered = state.get("summary_covered_count") or 0
    uncovered = messages[covered:]
    # `.text`, not `.content`: content can be a list of blocks.
    last_user_msg = next(
        (m.text for m in reversed(messages) if isinstance(m, HumanMessage)),
        "",
    )
    collection_id = state.get("collection_id")
    state_summary = state.get("summary")
    top_k = _requested_top_k(config)

    # 1. Query rewrite
    retrieval_query = await rewrite_query_for_retrieval(
        history=uncovered,
        current=last_user_msg,
        summary=state_summary,
    )

    # 2. Embedding + retrieval
    embedder = EmbeddingService()
    chunks: list[RetrievedChunk] = []
    retrieval_failed = False
    try:
        query_embedding = await embedder.embed(retrieval_query)
    except Exception:
        errors_total.labels(type="embedding").inc()
        logger.exception("embedding_failed", node="search_docs")
        retrieval_failed = True

    if not retrieval_failed:
        async with SessionLocal() as db:
            try:
                chunks = await retrieve_pipeline(
                    db=db,
                    query_text=retrieval_query,
                    query_embedding=query_embedding,
                    collection_id=collection_id,
                    top_k=top_k,
                )
            except Exception:
                errors_total.labels(type="retrieval").inc()
                logger.exception(
                    "retrieval_failed",
                    node="search_docs",
                    collection_id=collection_id,
                    top_k=top_k,
                )
                retrieval_failed = True

    # Best cosine among the results, not chunks[0].relevance: relevance is the
    # fused RRF score, relative by construction, while cosine says how close
    # the best match is in absolute terms. A failed retrieval, or one with no
    # cosine at all, records nothing: a 0.0 there would read as "irrelevant".
    cosines = [c.cosine for c in chunks if c.cosine is not None]
    top_rel = max(cosines) if cosines else None
    if not retrieval_failed:
        retrieval_chunks.observe(len(chunks))
        if top_rel is not None:
            retrieval_top_relevance.observe(top_rel)

    logger.info(
        "retrieval_done",
        chunks=len(chunks),
        top_relevance=top_rel,
        collection_id=collection_id,
        top_k=top_k,
        failed=retrieval_failed,
    )

    if retrieval_failed or not chunks:
        reply: FixedReply = "retrieval_failed" if retrieval_failed else "no_documents"
        return _turn_result(
            state,
            answer=fixed_reply(reply, messages),
            chunks=[],
            included_chunks=[],
            gen_md=_NO_GENERATION,
            extra_md={"retrieval_failed": retrieval_failed},
            collection_id=collection_id,
            top_rel=top_rel,
            failed=retrieval_failed,
        )

    # 3. Generation
    rag_user_prompt, included_chunks = build_user_prompt(last_user_msg, chunks)

    provider, model, temperature = llm_service_for("rag")
    history_for_llm = build_prompt_messages(state, exclude_last=True)
    history_for_llm.append({"role": "user", "content": rag_user_prompt})

    structured_md: dict = {}
    generation_failed = False

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
            answer = (parsed.answer or "").strip() or fixed_reply(
                "not_enough_information", messages
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
            answer = resp.text.strip() or fixed_reply("not_enough_information", messages)
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
        answer = fixed_reply("generation_failed", messages)
        gen_md = {"generation_model": model}
        generation_failed = True

    if (
        settings.agent.citation_semantic_validation
        and included_chunks
        and answer
        and not generation_failed
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

    return _turn_result(
        state,
        answer=answer,
        chunks=chunks,
        included_chunks=included_chunks,
        gen_md=gen_md,
        extra_md=structured_md,
        collection_id=collection_id,
        top_rel=top_rel,
        failed=generation_failed,
    )


def _turn_result(
    state: AgentState,
    *,
    answer: str,
    chunks: list[RetrievedChunk],
    included_chunks: list[RetrievedChunk],
    gen_md: dict,
    extra_md: dict,
    collection_id: int | None,
    top_rel: float | None,
    failed: bool = False,
) -> dict:
    """State update for the turn. A `failed` answer reports a failure: it
    cites nothing, the next prompts leave it out, and `/agent/chat` logs no
    query for it."""
    # The chunks were retrieved (`chunks_retrieved` still counts them), but
    # sources listed under "I had a problem…" would read as its sources.
    cited = [] if failed else included_chunks
    citations = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "document_title": c.document_title,
            "chunk_index": c.chunk_index,
            "relevance": c.relevance,
            "cosine": c.cosine,
            "text_preview": (c.text[:500] + "...") if len(c.text) > 500 else c.text,
        }
        for c in cited
    ]
    msg_metadata = {
        "model": gen_md.get("generation_model"),
        "tokens_input": gen_md.get("generation_tokens_input", 0),
        "tokens_output": gen_md.get("generation_tokens_output", 0),
        "latency_ms": gen_md.get("generation_latency_ms", 0),
        "chunks_retrieved": len(chunks),
    }
    prev = state.get("metadata") or {}
    kwargs = {"metadata": msg_metadata, "citations": citations}
    message = (
        failed_answer(answer, **kwargs)
        if failed
        else AIMessage(content=answer, additional_kwargs=kwargs)
    )
    return {
        "messages": [message],
        "retrieved_chunks": citations,
        "metadata": {
            **prev,
            **gen_md,
            **extra_md,
            "collection_id_active": collection_id,
            "retrieval_chunks_count": len(chunks),
            "retrieval_top_relevance": top_rel,
            "turn_failed": failed,
        },
    }
