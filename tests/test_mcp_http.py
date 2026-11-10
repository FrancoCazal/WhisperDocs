"""The key check in front of /mcp (audit M-2), without a database: the scoped
tools over HTTP are test_mcp_http_db.py's."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from app.config import settings
from app.main import app
from app.mcp.access import EVERYTHING, NOTHING, Access, KeyAuth, access_of, key_hash
from app.mcp.http import McpOverHttp, transport_security
from app.mcp.keys import _parser

MASTER = "the-api-key"


class _Db:
    """A session whose key lookup finds `row`."""

    def __init__(self, row=None) -> None:
        self.row = row
        self.lookups = 0

    async def scalar(self, _statement):
        self.lookups += 1
        return self.row


class _Inner:
    """The app behind KeyAuth: records the access it was given."""

    def __init__(self) -> None:
        self.scopes: list[dict] = []

    async def __call__(self, scope, receive, send):
        self.scopes.append(scope)
        if scope["type"] != "http":
            return
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _guard(db: _Db) -> tuple[KeyAuth, _Inner]:
    @asynccontextmanager
    async def sessions():
        yield db

    inner = _Inner()
    return KeyAuth(inner, sessions), inner


async def _post(asgi, headers: dict[str, str] | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=asgi)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as c:
        return await c.post("/mcp", json={}, headers=headers or {})


@pytest.fixture
def master(monkeypatch):
    monkeypatch.setattr(settings, "api_key", SecretStr(MASTER))


async def test_without_the_api_key_no_key_reaches_everything(monkeypatch):
    monkeypatch.setattr(settings, "api_key", None)
    db = _Db()
    guard, inner = _guard(db)

    response = await _post(guard)

    assert response.status_code == 200
    assert access_of(SimpleNamespace(scope=inner.scopes[0])) == EVERYTHING
    assert db.lookups == 0


@pytest.mark.usefixtures("master")
@pytest.mark.parametrize(
    "headers",
    [{"Authorization": f"Bearer {MASTER}"}, {"authorization": f"bearer  {MASTER} "}],
    ids=["bearer", "any-case"],
)
async def test_the_api_key_reaches_everything(headers):
    db = _Db()
    guard, inner = _guard(db)

    response = await _post(guard, headers)

    assert response.status_code == 200
    assert access_of(SimpleNamespace(scope=inner.scopes[0])) == EVERYTHING
    assert db.lookups == 0


@pytest.mark.usefixtures("master")
async def test_the_api_key_header_works_too():
    guard, inner = _guard(_Db())

    response = await _post(guard, {"X-API-Key": MASTER})

    assert response.status_code == 200
    assert access_of(SimpleNamespace(scope=inner.scopes[0])) == EVERYTHING


@pytest.mark.usefixtures("master")
async def test_a_stored_key_reaches_its_collections():
    row = SimpleNamespace(name="claude-code", collection_ids=[3, 5])
    db = _Db(row)
    guard, inner = _guard(db)

    every = SimpleNamespace(name="all", collection_ids=None)
    response = await _post(guard, {"Authorization": "Bearer wdk_abc"})
    guard_all, inner_all = _guard(_Db(every))
    await _post(guard_all, {"X-API-Key": "wdk_def"})

    assert response.status_code == 200
    assert access_of(SimpleNamespace(scope=inner.scopes[0])) == Access(
        "claude-code", frozenset({3, 5})
    )
    assert access_of(SimpleNamespace(scope=inner_all.scopes[0])) == Access("all", None)
    assert db.lookups == 1


@pytest.mark.usefixtures("master")
@pytest.mark.parametrize(
    ("headers", "code"),
    [
        ({}, "api_key_required"),
        ({"Authorization": "Basic abc"}, "api_key_required"),
        ({"Authorization": "Bearer  "}, "api_key_required"),
        ({"Authorization": "Bearer wdk_unknown"}, "api_key_invalid"),
        ({"X-API-Key": "wdk_unknown"}, "api_key_invalid"),
    ],
)
async def test_no_valid_key_is_a_401(headers, code):
    guard, inner = _guard(_Db())

    response = await _post(guard, headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["code"] == code
    assert inner.scopes == []


async def test_other_scopes_pass_through():
    guard, inner = _guard(_Db())

    await guard({"type": "lifespan"}, None, None)

    assert inner.scopes == [{"type": "lifespan"}]


def test_a_request_without_keyauth_reaches_nothing():
    assert access_of(None) == EVERYTHING
    assert access_of(SimpleNamespace(scope={"type": "http"})) == NOTHING
    assert access_of(SimpleNamespace(scope={"state": {"mcp_access": "forged"}})) == NOTHING
    assert not NOTHING.reaches(1)


def test_only_the_hash_of_a_key_is_kept():
    assert key_hash("wdk_x") == key_hash("wdk_x") != key_hash("wdk_y")
    assert len(key_hash("wdk_x")) == 64


async def test_mcp_answers_503_outside_the_lifespan(monkeypatch):
    monkeypatch.setattr(settings, "api_key", None)

    @asynccontextmanager
    async def sessions():
        yield _Db()

    response = await _post(McpOverHttp(sessions=sessions))

    assert response.status_code == 503
    assert response.json()["code"] == "mcp_down"


async def test_the_api_serves_mcp_behind_the_key(master):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as c:
        response = await c.post("/mcp", json={})
        schema = (await c.get("/openapi.json")).json()

    assert response.status_code == 401
    assert response.json()["code"] == "api_key_required"
    assert "/mcp" not in schema["paths"]


def test_the_allowed_hosts_give_the_allowed_origins():
    security = transport_security(["localhost:*", "docs.example.com"])

    assert security.enable_dns_rebinding_protection
    assert security.allowed_hosts == ["localhost:*", "docs.example.com"]
    assert security.allowed_origins == [
        "http://localhost:*",
        "https://localhost:*",
        "http://docs.example.com",
        "https://docs.example.com",
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["create", "k"],
        ["create", "k", "--all", "--collection", "1"],
        ["create", "k", "--collection", "x"],
        ["revoke"],
    ],
)
def test_a_key_reaches_everything_only_when_asked(argv):
    with pytest.raises(SystemExit):
        _parser().parse_args(argv)


def test_the_key_command_reads_its_collections():
    args = _parser().parse_args(["create", "k", "--collection", "3", "--collection", "5"])
    every = _parser().parse_args(["create", "k", "--all"])

    assert (args.name, args.collections, args.all) == ("k", [3, 5], False)
    assert (every.collections, every.all) == (None, True)
