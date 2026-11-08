from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Collection, Document
from app.schemas.collection import CollectionCreate, CollectionRead

router = APIRouter(prefix="/collections", tags=["collections"])


def _with_totals() -> Select:
    """Collections with their document, chunk and token totals."""
    totals = (
        select(
            Document.collection_id,
            func.count(Document.id).label("documents"),
            func.sum(Document.chunk_count).label("chunks"),
            func.sum(Document.total_tokens).label("tokens"),
        )
        .group_by(Document.collection_id)
        .subquery()
    )
    return select(
        Collection,
        func.coalesce(totals.c.documents, 0),
        func.coalesce(totals.c.chunks, 0),
        func.coalesce(totals.c.tokens, 0),
    ).outerjoin(totals, totals.c.collection_id == Collection.id)


def _read(row) -> CollectionRead:
    collection, documents, chunks, tokens = row
    return CollectionRead.model_validate(collection).model_copy(
        update={
            "document_count": int(documents),
            "chunk_count": int(chunks),
            "total_tokens": int(tokens),
        }
    )


@router.post(
    "",
    operation_id="create_collection",
    response_model=CollectionRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_collection(
    payload: CollectionCreate,
    db: AsyncSession = Depends(get_db),
) -> Collection:
    collection = Collection(name=payload.name, description=payload.description)
    db.add(collection)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Collection with name '{payload.name}' already exists",
        )
    await db.refresh(collection)
    return collection


@router.get("", operation_id="list_collections", response_model=list[CollectionRead])
async def list_collections(db: AsyncSession = Depends(get_db)) -> list[CollectionRead]:
    result = await db.execute(_with_totals().order_by(Collection.created_at.desc()))
    return [_read(row) for row in result.all()]


@router.get("/{collection_id}", operation_id="get_collection", response_model=CollectionRead)
async def get_collection(
    collection_id: int,
    db: AsyncSession = Depends(get_db),
) -> CollectionRead:
    row = (await db.execute(_with_totals().where(Collection.id == collection_id))).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
    return _read(row)


@router.delete(
    "/{collection_id}", operation_id="delete_collection", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_collection(
    collection_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    collection = await db.get(Collection, collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
    await db.delete(collection)
    await db.commit()
