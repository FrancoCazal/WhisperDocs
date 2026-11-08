"""A single-turn answer with its citations: retrieval, then one generation.

Behind `POST /query` and the MCP server's `ask_documents` (audit M-1), so both
answer a question alike: the same search, the same prompt, the same model, and
a row in `query_logs` either way.
"""

from __future__ import annotations

import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import QueryLog
from app.schemas.query import Citation, QueryMetadata, QueryRequest, QueryResponse
from app.services.embeddings import EmbeddingProvider
from app.services.llm import llm_service_for
from app.services.llm.pricing import estimate_cost_usd
from app.services.rag import RAG_SYSTEM_PROMPT, build_user_prompt
from app.services.retrieval_pipeline import retrieve as retrieve_pipeline

NO_DOCUMENTS_ANSWER = "I don't have any indexed documents to answer this question."


class GenerationError(Exception):
    """The model call failed. The provider's exception is the `__cause__`."""


async def answer_question(
    db: AsyncSession, embedder: EmbeddingProvider, request: QueryRequest
) -> QueryResponse:
    """Answer `request.question` from the indexed documents, and log it.

    Raises the embedder's `EmbeddingError`, or `GenerationError` when the
    model call fails.
    """
    start = time.perf_counter()
    provider, model, temperature = llm_service_for("rag")

    question_embedding = await embedder.embed(request.question)
    chunks = await retrieve_pipeline(
        db=db,
        query_text=request.question,
        query_embedding=question_embedding,
        collection_id=request.collection_id,
        top_k=request.top_k,
    )

    if not chunks:
        latency_ms = int((time.perf_counter() - start) * 1000)
        db.add(
            QueryLog(
                question=request.question,
                answer=NO_DOCUMENTS_ANSWER,
                collection_id=request.collection_id,
                chunks_retrieved=0,
                model_used=model,
                tokens_input=0,
                tokens_output=0,
                latency_ms=latency_ms,
                session_id=request.session_id,
            )
        )
        await db.commit()
        return QueryResponse(
            answer=NO_DOCUMENTS_ANSWER,
            citations=[],
            metadata=QueryMetadata(
                model=model,
                tokens_input=0,
                tokens_output=0,
                latency_ms=latency_ms,
                chunks_retrieved=0,
            ),
        )

    # The citations index into the chunks that fit the context budget.
    user_prompt, included = build_user_prompt(request.question, chunks)

    try:
        llm_response = await provider.generate(
            system=RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            model=model,
            temperature=temperature,
            max_tokens=1024,
        )
    except Exception as exc:
        # The providers raise their SDK's exceptions, not LLMError: catching
        # that alone let every model failure through as a bare 500.
        raise GenerationError(f"Generation with {model} failed") from exc

    latency_ms = int((time.perf_counter() - start) * 1000)
    cost_usd = estimate_cost_usd(
        llm_response.model, llm_response.tokens_input, llm_response.tokens_output
    )

    db.add(
        QueryLog(
            question=request.question,
            answer=llm_response.text,
            collection_id=request.collection_id,
            chunks_retrieved=len(chunks),
            model_used=llm_response.model,
            tokens_input=llm_response.tokens_input,
            tokens_output=llm_response.tokens_output,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            session_id=request.session_id,
        )
    )
    await db.commit()

    return QueryResponse(
        answer=llm_response.text,
        citations=[
            Citation(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_title=c.document_title,
                chunk_index=c.chunk_index,
                relevance=c.relevance,
                cosine=c.cosine,
            )
            for c in included
        ],
        metadata=QueryMetadata(
            model=llm_response.model,
            tokens_input=llm_response.tokens_input,
            tokens_output=llm_response.tokens_output,
            latency_ms=latency_ms,
            chunks_retrieved=len(chunks),
            cost_usd=cost_usd,
        ),
    )
