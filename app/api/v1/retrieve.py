"""The chunks for a query, without an answer (audit R20).

What another agent needs from a knowledge base: the passages and where they
come from, to answer with its own model. The search lives in
`app/services/search.py`, shared with the MCP server's `search_documents`.
It pays one embedding, so it counts against the questions rate limit.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import upstream_error
from app.api.rate_limit import questions_limit
from app.db import get_db
from app.logging_config import get_logger
from app.schemas.retrieve import RetrieveRequest, RetrieveResponse
from app.services.embedder import EmbeddingError, EmbeddingService, get_embedder
from app.services.search import OutOfScope, ScopeNotFound, search

logger = get_logger(__name__)

router = APIRouter(tags=["retrieval"])


@router.post(
    "/retrieve",
    operation_id="retrieve_chunks",
    response_model=RetrieveResponse,
    dependencies=[Depends(questions_limit)],
)
async def retrieve_chunks(
    payload: RetrieveRequest,
    db: AsyncSession = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder),
) -> RetrieveResponse:
    """The chunks that best match `query`, most relevant first, with their text
    and their document: no answer is generated. Scope it with `collection_id`,
    or with `document_id` for a single document."""
    try:
        return await search(db, embedder, payload)
    except ScopeNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except OutOfScope as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except EmbeddingError as exc:
        logger.exception("retrieve_embedding_failed")
        raise upstream_error(exc, "Embedding the query") from exc
