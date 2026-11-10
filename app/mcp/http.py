"""The MCP server over Streamable HTTP, at the API's `/mcp` (audit M-1, M-2).

Stateless, as the 2026-07-28 spec has it: each POST is a whole exchange, with
a JSON answer, so the API keeps no MCP session between requests. `KeyAuth`
checks the caller's key first (`app/mcp/access.py`); then the SDK checks the
Host header against `MCP_ALLOWED_HOSTS`, its guard against DNS rebinding.

The SDK's session manager runs once per instance, so each run of the API's
lifespan builds a new one (`running`). Outside it, `/mcp` answers 503.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import Receive, Scope, Send

from app.config import settings
from app.db import SessionLocal
from app.mcp.access import KeyAuth, Sessions
from app.mcp.server import WhisperDocsServer, build_server

MCP_PATH = "/mcp"


def transport_security(hosts: list[str]) -> TransportSecuritySettings:
    origins = [f"{scheme}://{host}" for host in hosts for scheme in ("http", "https")]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins
    )


class McpOverHttp:
    """The ASGI endpoint at `/mcp`: the key, then the SDK's transport."""

    def __init__(
        self, server: WhisperDocsServer | None = None, sessions: Sessions = SessionLocal
    ) -> None:
        self.server = server or build_server(sessions=sessions)
        self._running = False
        self._auth = KeyAuth(self._transport, sessions)

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        self.server.streamable_http_app(
            streamable_http_path=MCP_PATH,
            stateless_http=True,
            json_response=True,
            transport_security=transport_security(settings.mcp_allowed_hosts),
        )
        async with self.server.session_manager.run():
            self._running = True
            try:
                yield
            finally:
                self._running = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._auth(scope, receive, send)

    async def _transport(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._running:
            await _not_running(send)
            return
        await self.server.session_manager.handle_request(scope, receive, send)


async def _not_running(send: Send) -> None:
    body = json.dumps({"detail": "The MCP server is not running.", "code": "mcp_down"}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 503,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
