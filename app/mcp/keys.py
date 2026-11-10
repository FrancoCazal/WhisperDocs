"""`python -m app.mcp.keys`: the keys of the MCP server over HTTP (audit M-2).

    python -m app.mcp.keys create NAME --collection 3 [--collection 5]
    python -m app.mcp.keys create NAME --all
    python -m app.mcp.keys list
    python -m app.mcp.keys revoke NAME

A key reaches the collections it was made for, or every one with `--all`;
one of the two is required, so a key reaches everything only when asked to.
`create` prints the key once: only its SHA-256 is stored. Under compose:
`docker compose exec api python -m app.mcp.keys ...`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.mcp.access import key_hash, new_key
from app.models import ApiKey, Collection

# The key's first characters, kept to tell keys apart in `list`.
PREFIX_LENGTH = 12


class KeyCommandError(Exception):
    """A key that cannot be made or revoked, and why."""


@dataclass(frozen=True)
class KeyInfo:
    name: str
    prefix: str
    collection_ids: list[int] | None
    created_at: datetime
    revoked_at: datetime | None


async def create_key(db: AsyncSession, name: str, collection_ids: Sequence[int] | None) -> str:
    """Store a new key and return it. `collection_ids` None reaches every
    collection."""
    ids = None if collection_ids is None else sorted(set(collection_ids))
    if ids is not None:
        if not ids:
            raise KeyCommandError("A key needs a collection, or --all.")
        found = set(await db.scalars(select(Collection.id).where(Collection.id.in_(ids))))
        missing = [c for c in ids if c not in found]
        if missing:
            raise KeyCommandError(f"No collection {', '.join(map(str, missing))}.")
    key = new_key()
    db.add(
        ApiKey(name=name, key_hash=key_hash(key), prefix=key[:PREFIX_LENGTH], collection_ids=ids)
    )
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise KeyCommandError(f"A key named {name!r} exists already, revoked or not.") from exc
    return key


async def list_keys(db: AsyncSession) -> list[KeyInfo]:
    rows = await db.scalars(select(ApiKey).order_by(ApiKey.created_at, ApiKey.id))
    return [KeyInfo(r.name, r.prefix, r.collection_ids, r.created_at, r.revoked_at) for r in rows]


async def revoke_key(db: AsyncSession, name: str) -> None:
    revoked = await db.scalar(
        update(ApiKey)
        .where(ApiKey.name == name, ApiKey.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
        .returning(ApiKey.id)
    )
    if revoked is None:
        raise KeyCommandError(f"No key named {name!r} that is not revoked already.")
    await db.commit()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.mcp.keys", description="The MCP server's keys over HTTP."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Make a key and print it, once.")
    create.add_argument("name", help="Who or what uses the key, unique.")
    reach = create.add_mutually_exclusive_group(required=True)
    reach.add_argument(
        "--collection",
        type=int,
        action="append",
        dest="collections",
        metavar="ID",
        help="A collection the key reaches; repeat for more.",
    )
    reach.add_argument("--all", action="store_true", help="Every collection, today's and later.")
    commands.add_parser("list", help="The keys, without the keys themselves.")
    revoke = commands.add_parser("revoke", help="Revoke a key, at once.")
    revoke.add_argument("name")
    return parser


def _row(info: KeyInfo) -> str:
    reach = "all" if info.collection_ids is None else ",".join(map(str, info.collection_ids))
    state = f"revoked {info.revoked_at:%Y-%m-%d}" if info.revoked_at else "active"
    return f"{info.name}\t{info.prefix}…\tcollections={reach}\t{state}"


async def _run(args: argparse.Namespace) -> int:
    from app.db import SessionLocal

    async with SessionLocal() as db:
        try:
            if args.command == "create":
                key = await create_key(db, args.name, None if args.all else args.collections)
                print(key)
                print("Shown once: only its hash is stored.", file=sys.stderr)
            elif args.command == "list":
                for info in await list_keys(db):
                    print(_row(info))
            else:
                await revoke_key(db, args.name)
                print(f"Revoked {args.name!r}.", file=sys.stderr)
        except KeyCommandError as exc:
            print(exc, file=sys.stderr)
            return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if sys.platform == "win32":
        # asyncpg and the Proactor loop: see app/main.py.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
