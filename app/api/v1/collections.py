from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models import Collection
from app.schemas.collection import CollectionCreate, CollectionRead

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionRead, status_code=status.HTTP_201_CREATED)
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


@router.get("", response_model=list[CollectionRead])
async def list_collections(db: AsyncSession = Depends(get_db)) -> list[Collection]:
    result = await db.execute(select(Collection).order_by(Collection.created_at.desc()))
    return list(result.scalars().all())


@router.get("/{collection_id}", response_model=CollectionRead)
async def get_collection(
    collection_id: int,
    db: AsyncSession = Depends(get_db),
) -> Collection:
    collection = await db.get(Collection, collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
    return collection


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    collection = await db.get(Collection, collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
    await db.delete(collection)
    await db.commit()
