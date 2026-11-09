"""Scan the stored chunks again for text that reads as an instruction to the model.

Ingestion flags each chunk it stores (app/services/instruction_scan.py). A
chunk stored before the flags existed has none, and one stored before a rule
changed has that day's. This sets every chunk's flags from the current rules,
and each document's count. No model is called.

Usage:
    python -m scripts.flag_instructions                    # every collection
    python -m scripts.flag_instructions --collection-id 3

Prints a line for each document with flagged chunks, then the totals.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionLocal
from app.models import Chunk, Document
from app.services.instruction_scan import instruction_flags


@dataclass
class Scanned:
    document_id: int
    filename: str
    flagged_chunks: int
    # Chunks whose flags this run changed.
    changed: int


async def rescan(db: AsyncSession, *, collection_id: int | None = None) -> list[Scanned]:
    """Set the flags of every chunk in the collection (or in all of them),
    and each document's count, committing per document."""
    query = select(Document).order_by(Document.id)
    if collection_id is not None:
        query = query.where(Document.collection_id == collection_id)
    documents = list((await db.execute(query)).scalars().all())
    scanned: list[Scanned] = []
    for document in documents:
        chunks = (
            await db.execute(
                select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.chunk_index)
            )
        ).scalars()
        changed = flagged = 0
        for chunk in chunks:
            flags = instruction_flags(chunk.text)
            if flags != list(chunk.instruction_flags):
                chunk.instruction_flags = flags
                changed += 1
            flagged += bool(flags)
        document.flagged_chunks = flagged
        await db.commit()
        scanned.append(Scanned(document.id, document.filename, flagged, changed))
    return scanned


async def main_async(collection_id: int | None) -> int:
    async with SessionLocal() as db:
        scanned = await rescan(db, collection_id=collection_id)
    for s in scanned:
        if s.flagged_chunks:
            print(f"document {s.document_id} ({s.filename}): {s.flagged_chunks} flagged chunks")
    print(
        f"{len(scanned)} documents, {sum(s.flagged_chunks for s in scanned)} flagged chunks, "
        f"{sum(s.changed for s in scanned)} chunks changed"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan the stored chunks again for instruction-like text."
    )
    parser.add_argument("--collection-id", type=int, default=None)
    args = parser.parse_args()
    return asyncio.run(main_async(args.collection_id))


if __name__ == "__main__":
    raise SystemExit(main())
