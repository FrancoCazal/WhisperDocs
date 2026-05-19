"""Seed the eval corpus: ingest every doc under fixtures/seed-docs/ into a collection.

Usage:
    python -m scripts.seed_eval                            # collection name "Eval Golden"
    python -m scripts.seed_eval --collection-name MyName

Idempotent: skips files already ingested in the target collection (by filename).
Prints the resolved collection_id on stdout — pipe into the runner:

    COLL_ID=$(python -m scripts.seed_eval --quiet)
    python -m app.eval.runner run --collection-id "$COLL_ID" --markdown
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.logging_config import get_logger
from app.models import Chunk, Collection, Document
from app.services.chunker import TextChunker
from app.services.embedder import EmbeddingService
from app.services.lang import detect_language
from app.services.text_extractor import detect_file_type, extract_text

logger = get_logger(__name__)

DEFAULT_COLLECTION_NAME = "Eval Golden"
SEED_ROOT = Path("fixtures/seed-docs")


async def get_or_create_collection(db, name: str) -> Collection:
    result = await db.execute(select(Collection).where(Collection.name == name))
    coll = result.scalar_one_or_none()
    if coll is not None:
        return coll
    coll = Collection(name=name, description="Seed corpus for the eval suite.")
    db.add(coll)
    await db.commit()
    await db.refresh(coll)
    return coll


async def ingest_file(
    path: Path,
    collection_id: int,
    db,
    embedder: EmbeddingService,
) -> Document | None:
    existing = await db.execute(
        select(Document).where(
            Document.collection_id == collection_id,
            Document.filename == path.name,
        )
    )
    if existing.scalar_one_or_none() is not None:
        return None

    file_bytes = path.read_bytes()
    file_type = detect_file_type(path.name)
    raw_text = extract_text(file_bytes, file_type)

    chunker = TextChunker()
    chunks_data = chunker.chunk(raw_text)
    if not chunks_data:
        raise ValueError(f"{path.name}: produced no chunks")

    hashes = [hashlib.sha256(c.text.encode("utf-8")).hexdigest() for c in chunks_data]
    existing_q = (
        select(Chunk.text_hash)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.collection_id == collection_id)
        .where(Chunk.text_hash.in_(hashes))
    )
    skip_hashes = {row[0] for row in (await db.execute(existing_q)).all()}
    fresh = [
        (cd.index, cd.text, cd.token_count, h)
        for cd, h in zip(chunks_data, hashes, strict=True)
        if h not in skip_hashes
    ]
    if not fresh:
        return None

    embeddings = await embedder.embed_batch([t for _i, t, _tc, _h in fresh])

    document = Document(
        collection_id=collection_id,
        title=path.name,
        filename=path.name,
        file_type=file_type,
        raw_text=raw_text,
        chunk_count=len(fresh),
        total_tokens=sum(tc for _i, _t, tc, _h in fresh),
        language=detect_language(raw_text),
    )
    db.add(document)
    await db.flush()
    db.add_all(
        Chunk(
            document_id=document.id,
            chunk_index=idx,
            text=t,
            token_count=tc,
            text_hash=h,
            embedding=embedding,
        )
        for (idx, t, tc, h), embedding in zip(fresh, embeddings, strict=True)
    )
    await db.commit()
    await db.refresh(document)
    return document


async def main_async(collection_name: str, root: Path, quiet: bool) -> int:
    if not root.exists():
        print(f"Seed root not found: {root}", file=sys.stderr)
        return 1

    paths = sorted(p for p in root.rglob("*.md") if p.is_file())
    if not paths:
        print(f"No .md files under {root}", file=sys.stderr)
        return 1

    embedder = EmbeddingService()
    async with SessionLocal() as db:
        coll = await get_or_create_collection(db, collection_name)
        coll_id = coll.id
        coll_name = coll.name
        ingested = 0
        skipped = 0
        for p in paths:
            try:
                doc = await ingest_file(p, coll_id, db, embedder)
            except Exception as exc:
                await db.rollback()
                if not quiet:
                    print(f"  FAIL {p.name}: {exc}", file=sys.stderr)
                continue
            if doc is None:
                skipped += 1
                if not quiet:
                    print(f"  skip {p.name} (already ingested)")
            else:
                ingested += 1
                if not quiet:
                    print(f"  +    {p.name} ({doc.chunk_count} chunks)")

        if not quiet:
            print(
                f"\nCollection '{coll_name}' (id={coll_id}): "
                f"+{ingested} ingested, {skipped} skipped, total {len(paths)} files."
            )
        if quiet:
            print(coll_id)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the eval corpus collection.")
    parser.add_argument("--collection-name", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--root", type=Path, default=SEED_ROOT)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the collection id on stdout (for shell pipelines).",
    )
    args = parser.parse_args()
    code = asyncio.run(main_async(args.collection_name, args.root, args.quiet))
    sys.exit(code)


if __name__ == "__main__":
    main()
