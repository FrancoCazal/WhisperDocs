"""HTTP rate, errors and duration for Prometheus (audit E13).

A pure ASGI middleware: it reads the status from the response's start and
times the request until the app returns, so a stream counts until its last
event.

The route label is the matched route's template as the OpenAPI schema has
it (`/api/v1/documents/{document_id}`). FastAPI leaves the matched route in
the scope, but (in 0.162) with the path it was declared with, relative to
its router's prefix (`/documents/{document_id:int}`), so the template comes
from the schema by the route's operation id. A path no route matched is
"unmatched", so a scanner's paths do not become label values; a body refused
by its declared size is answered before routing, so its 413 is "unmatched"
too. The method label is bounded the same way.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.metrics import http_request_duration_seconds, http_requests_total

logger = logging.getLogger(__name__)

UNMATCHED = "unmatched"
# The client went away before the answer started (nginx's code for it).
CLIENT_CLOSED = 499
_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


def _templates(app: Any) -> dict[str, str]:
    """Each operation id's path template, from the app's OpenAPI schema."""
    try:
        paths = app.openapi()["paths"]
    except Exception:
        logger.warning("http_metrics_no_openapi", exc_info=True)
        return {}
    return {
        operation["operationId"]: path
        for path, item in paths.items()
        for operation in item.values()
        if isinstance(operation, dict) and "operationId" in operation
    }


class HTTPMetricsMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._templates: dict[str, str] | None = None

    def _route(self, scope: Scope) -> str:
        route = scope.get("route")
        if route is None:
            return UNMATCHED
        operation = getattr(route, "operation_id", None) or getattr(route, "unique_id", None)
        if operation is not None:
            if self._templates is None:
                self._templates = _templates(scope.get("app"))
            template = self._templates.get(operation)
            if template:
                return template
        # A route outside the schema (FastAPI's own /docs): its declared path.
        path = getattr(route, "path_format", None)
        return path if isinstance(path, str) and path else UNMATCHED

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        status: int | None = None

        async def send_with_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        started = time.perf_counter()
        try:
            await self.app(scope, receive, send_with_status)
        except asyncio.CancelledError:
            if status is None:
                status = CLIENT_CLOSED
            raise
        except Exception:
            # A stream that fails after its start keeps the status it sent.
            if status is None:
                status = 500
            raise
        finally:
            method = scope["method"] if scope["method"] in _METHODS else "OTHER"
            route = self._route(scope)
            http_requests_total.labels(method=method, route=route, status=str(status or 500)).inc()
            http_request_duration_seconds.labels(method=method, route=route).observe(
                time.perf_counter() - started
            )
