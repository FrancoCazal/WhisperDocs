"""Request body size limits per path (audit R8).

Starlette parses a multipart form, spooling every file to disk, before the
endpoint runs. A size check inside the endpoint therefore comes after the
whole upload was received. This middleware answers 413 before that: up front
from Content-Length, or, for a request without one (chunked), as soon as the
bytes read pass the limit.
"""

from __future__ import annotations

from fastapi import HTTPException, status
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import error_body


def _too_large(limit: int) -> str:
    return f"Request body exceeds the {limit // (1024 * 1024)} MB limit for this endpoint"


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, limits: dict[str, int]) -> None:
        self.app = app
        self.limits = limits

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = self.limits.get(scope["path"]) if scope["type"] == "http" else None
        if limit is None:
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            response = JSONResponse(
                error_body(scope, 413, "payload_too_large", _too_large(limit)),
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # Raised while the endpoint parses the form; FastAPI passes
                    # HTTPException through, and it becomes the 413 response.
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=_too_large(limit),
                    )
            return message

        await self.app(scope, limited_receive, send)
