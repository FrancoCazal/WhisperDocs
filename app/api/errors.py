"""The API's errors: a stable `code`, a message without internals, and the
request id that finds the details in the logs (audit R9, A9).

Every error response has the same body:

    {"detail": "...", "code": "not_found", "request_id": "3f2a..."}

`detail` is still what FastAPI clients read: a string, or for a 422 the list
of validation errors. `code` is what a client branches on; the message can
change. A 502 from a failed model or embedding call adds `kind`, from
`classify_error` (billing, auth, rate_limit, ...). A 5xx message ends with the
request id, so a user can quote it.

`RequestContextMiddleware` gives each request an id (the caller's
`X-Request-ID` when it is a plain token, a new one otherwise), returns it in the
`X-Request-ID` header, and binds it to the log context: every log line of the
request carries it. It also turns an exception nothing handled into a 500 with
that body. Registered as an `Exception` handler, Starlette would answer it from
outside the CORS middleware, and the browser would report a CORS error instead
of the 500.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.logging_config import get_logger
from app.services.llm.errors import KIND_TEXT, classify_error

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_INCOMING_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# The code of an HTTPException raised without one, by its status.
STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "unprocessable",
    429: "rate_limited",
    500: "internal_error",
    502: "upstream_error",
    503: "unavailable",
}


class APIError(Exception):
    """An error response with its own `code` and a message fit for clients."""

    def __init__(
        self,
        status_code: int,
        code: str,
        detail: str,
        *,
        headers: dict[str, str] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.headers = headers
        self.extra = extra


def upstream_error(exc: BaseException, what: str) -> APIError:
    """A 502 for a failed model or embedding call: "<what> failed: <why>.",
    with the provider's own message left to the logs."""
    kind = classify_error(exc)
    return APIError(
        status.HTTP_502_BAD_GATEWAY,
        "upstream_error",
        f"{what} failed: {KIND_TEXT[kind]}.",
        kind=kind,
    )


def request_id(scope: Scope) -> str | None:
    return (scope.get("state") or {}).get("request_id")


def error_body(scope: Scope, status_code: int, code: str, detail: Any, **extra: Any) -> dict:
    rid = request_id(scope)
    if status_code >= 500 and isinstance(detail, str) and rid:
        detail = f"{detail} Request id {rid}."
    return {"detail": detail, "code": code, "request_id": rid, **extra}


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, APIError)
    body = error_body(request.scope, exc.status_code, exc.code, exc.detail, **exc.extra)
    return JSONResponse(body, status_code=exc.status_code, headers=exc.headers)


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = STATUS_CODES.get(exc.status_code, "error")
    body = error_body(request.scope, exc.status_code, code, exc.detail)
    return JSONResponse(body, status_code=exc.status_code, headers=exc.headers)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    body = error_body(request.scope, 422, "validation_error", jsonable_encoder(exc.errors()))
    return JSONResponse(body, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(APIError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        rid = incoming if _INCOMING_ID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid
        started = False

        async def send_with_id(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = [*message.get("headers", []), (b"x-request-id", rid.encode())]
                message = {**message, "headers": headers}
            await send(message)

        with structlog.contextvars.bound_contextvars(request_id=rid):
            try:
                await self.app(scope, receive, send_with_id)
            except Exception:
                logger.exception("unhandled_error", method=scope["method"], path=scope["path"])
                if started:
                    # A stream already sent its headers: nothing to answer with.
                    raise
                body = error_body(scope, 500, "internal_error", "Internal error.")
                await JSONResponse(body, status_code=500)(scope, receive, send_with_id)
