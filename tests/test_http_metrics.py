"""HTTP metrics (audit E13): each request counted by method, route template
and status, and timed until the response's last byte."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import REGISTRY

from app.api.body_limit import BodySizeLimitMiddleware
from app.api.errors import RequestContextMiddleware
from app.api.http_metrics import HTTPMetricsMiddleware


def _count(method: str, route: str, status: str) -> float:
    labels = {"method": method, "route": route, "status": status}
    return REGISTRY.get_sample_value("whisperdocs_http_requests_total", labels) or 0.0


def _timed(method: str, route: str) -> tuple[float, float]:
    labels = {"method": method, "route": route}
    name = "whisperdocs_http_request_duration_seconds"
    return (
        REGISTRY.get_sample_value(f"{name}_count", labels) or 0.0,
        REGISTRY.get_sample_value(f"{name}_sum", labels) or 0.0,
    )


@pytest.fixture
def client():
    app = FastAPI()

    @app.get("/things/{thing_id}")
    async def thing(thing_id: int):
        return {"id": thing_id}

    @app.get("/boom")
    async def boom():
        raise RuntimeError("boom")

    @app.get("/slow/stream")
    async def stream():
        async def body():
            yield b"a"
            await asyncio.sleep(0.05)
            yield b"b"

        return StreamingResponse(body())

    # As in app.main: the request context inside, turning an exception into
    # a 500 that the metrics see.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(HTTPMetricsMiddleware)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_requests_count_by_route_template_not_by_path(client):
    route = "/things/{thing_id}"
    ok, invalid = _count("GET", route, "200"), _count("GET", route, "422")

    async with client:
        await client.get("/things/1")
        await client.get("/things/2")
        await client.get("/things/abc")

    assert _count("GET", route, "200") == ok + 2
    assert _count("GET", route, "422") == invalid + 1


async def test_a_path_no_route_matches_is_unmatched(client):
    before = _count("GET", "unmatched", "404")

    async with client:
        await client.get("/wp-admin/setup-config.php")

    assert _count("GET", "unmatched", "404") == before + 1


async def test_an_unknown_method_is_other(client):
    before = _count("OTHER", "/things/{thing_id}", "405")

    async with client:
        await client.request("BREW", "/things/1")

    assert _count("OTHER", "/things/{thing_id}", "405") == before + 1


async def test_an_unhandled_error_counts_as_the_500_it_answers(client):
    before = _count("GET", "/boom", "500")

    async with client:
        response = await client.get("/boom")

    assert response.status_code == 500
    assert _count("GET", "/boom", "500") == before + 1


async def test_a_stream_is_timed_until_its_last_byte(client):
    count, seconds = _timed("GET", "/slow/stream")

    async with client:
        response = await client.get("/slow/stream")

    assert response.content == b"ab"
    new_count, new_seconds = _timed("GET", "/slow/stream")
    assert new_count == count + 1
    assert new_seconds - seconds >= 0.05


async def _call(app, scope_method: str = "GET") -> None:
    scope = {"type": "http", "method": scope_method, "path": "/x", "headers": []}

    async def send(_message):
        return None

    await HTTPMetricsMiddleware(app)(scope, None, send)


async def test_a_client_gone_before_the_answer_counts_as_499():
    async def cancelled(scope, receive, send):
        raise asyncio.CancelledError

    before = _count("GET", "unmatched", "499")

    with pytest.raises(asyncio.CancelledError):
        await _call(cancelled)

    assert _count("GET", "unmatched", "499") == before + 1


async def test_an_error_keeps_the_status_already_sent_or_counts_500():
    async def cut_stream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("cut mid-stream")

    async def failed(scope, receive, send):
        raise RuntimeError("before the answer")

    sent, unanswered = _count("GET", "unmatched", "200"), _count("GET", "unmatched", "500")

    with pytest.raises(RuntimeError):
        await _call(cut_stream)
    with pytest.raises(RuntimeError):
        await _call(failed)

    assert _count("GET", "unmatched", "200") == sent + 1
    assert _count("GET", "unmatched", "500") == unanswered + 1


async def test_the_app_counts_every_request_inside_cors_and_outside_its_context(monkeypatch):
    from app.config import settings
    from app.main import API_PREFIX, app

    monkeypatch.setattr(settings, "api_key", None)

    stack = [m.cls for m in app.user_middleware]
    # Outermost first: CORS answers preflights before the metrics, and the
    # request context turns an exception into a 500 inside them.
    assert stack == [
        CORSMiddleware,
        HTTPMetricsMiddleware,
        RequestContextMiddleware,
        BodySizeLimitMiddleware,
    ]
    route = f"{API_PREFIX}/metrics"
    # Declared as "/{collection_id}" on its router; the label is the full
    # template, from the OpenAPI schema.
    collection = f"{API_PREFIX}/collections/{{collection_id}}"
    before = _count("GET", route, "200")
    invalid = _count("GET", collection, "422")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(route)
        await client.get(f"{API_PREFIX}/collections/abc")

    assert response.status_code == 200
    assert "whisperdocs_http_requests_total" in response.text
    assert _count("GET", route, "200") == before + 1
    assert _count("GET", collection, "422") == invalid + 1
