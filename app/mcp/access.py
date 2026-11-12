"""Who may call the MCP server over HTTP, and which collections they reach
(audit M-2).

A request to `/mcp` brings a key as `Authorization: Bearer <key>` or in the
API's `X-API-Key` header:

- a key from `api_keys` reaches its collections, or all of them;
- the API's own `API_KEY` reaches all of them;
- no key: all of them while `API_KEY` is unset, as the REST API is open
  then; a 401 once it is set.

A revoked or unknown key is a 401 either way. `KeyAuth` checks the key and
puts its `Access` in the request's state; each tool reads it from the HTTP
request its message came in (`access_of`). Over stdio there is no request,
and the process has what the API has: every collection.

Writing (BE-36) takes more: a key from `api_keys` writes only when it was made
with `--write`. The API's own key, an open API and stdio write, as the REST
API lets them.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from secrets import compare_digest
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.types import ASGIApp, Receive, Scope, Send

from app.api.auth import API_KEY_HEADER, configured_key
from app.db import SessionLocal
from app.models import ApiKey

Sessions = Callable[[], AbstractAsyncContextManager[AsyncSession]]

KEY_PREFIX = "wdk_"
# Where KeyAuth leaves the caller's access in the request's state.
ACCESS_STATE = "mcp_access"


@dataclass(frozen=True)
class Access:
    """What a caller reaches. `collections` None is every collection.
    `can_write`: whether it may add, delete and move documents there."""

    key: str | None
    collections: frozenset[int] | None
    can_write: bool = False

    def reaches(self, collection_id: int) -> bool:
        return self.collections is None or collection_id in self.collections


EVERYTHING = Access(key=None, collections=None, can_write=True)
# For a request that reached a tool without KeyAuth: it gets nothing.
NOTHING = Access(key=None, collections=frozenset())


def new_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def access_for(db: AsyncSession, key: str | None) -> Access | None:
    """The access a key gives, or None when it gives none."""
    master = configured_key()
    if not key:
        return EVERYTHING if master is None else None
    if master is not None and compare_digest(key.encode(), master.encode()):
        return EVERYTHING
    row = await db.scalar(
        select(ApiKey).where(ApiKey.key_hash == key_hash(key), ApiKey.revoked_at.is_(None))
    )
    if row is None:
        return None
    collections = None if row.collection_ids is None else frozenset(row.collection_ids)
    return Access(key=row.name, collections=collections, can_write=row.can_write)


def _key_from(scope: Scope) -> str | None:
    headers = {name.decode("latin-1").lower(): value for name, value in scope["headers"]}
    auth = headers.get("authorization", b"").decode("latin-1")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    key = headers.get(API_KEY_HEADER.lower(), b"").decode("latin-1").strip()
    return key or None


class KeyAuth:
    """ASGI middleware in front of the MCP app: a 401 without a valid key."""

    def __init__(self, app: ASGIApp, sessions: Sessions = SessionLocal) -> None:
        self.app = app
        self.sessions = sessions

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        key = _key_from(scope)
        async with self.sessions() as db:
            access = await access_for(db, key)
        if access is None:
            code = "api_key_invalid" if key else "api_key_required"
            message = (
                "The key is not valid."
                if key
                else "This server needs a key: Authorization: Bearer <key>."
            )
            await _unauthorized(send, code, message)
            return
        scope.setdefault("state", {})[ACCESS_STATE] = access
        await self.app(scope, receive, send)


async def _unauthorized(send: Send, code: str, message: str) -> None:
    body = json.dumps({"detail": message, "code": code}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b"Bearer"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def access_of(request: Any) -> Access:
    """The access of the HTTP request a message came in; everything over
    stdio, where there is none."""
    if request is None:
        return EVERYTHING
    state = getattr(request, "scope", {}).get("state") or {}
    access = state.get(ACCESS_STATE)
    return access if isinstance(access, Access) else NOTHING


def actor_of(access: Access, request: Any) -> str:
    """Who an audit row names (app/models/audit_log.py): the key's name, the
    API's own key, the open API, or the stdio process."""
    if request is None:
        return "stdio"
    if access.key is not None:
        return access.key
    return "open" if configured_key() is None else "api-key"
