"""Keep every test off the network, except the test database.

Each test stubs the provider it calls, and a test that forgot to went to the
real one: with K_RECENT=2 in the dev .env, a summarize test sent its summary
to Bedrock. Every host name a test resolves is checked here. A host other than
this machine or the test database is refused, and the test fails even if the
code under test caught the error.

Host names are what this checks: a connection to a literal IP address in an
asyncio client does not resolve anything, and is not caught.
"""

from __future__ import annotations

import asyncio
import os
import selectors
import socket
import sys
from urllib.parse import urlsplit

import pytest

# Without credentials in the environment (as in CI), botocore asks the EC2
# metadata endpoint for them when a client is created.
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
# Spans stay in the process: an endpoint in the dev .env would send every
# test's spans to the trace viewer (app/telemetry.py). Set before `settings`
# is first imported; the environment wins over .env.
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = ""

_real_getaddrinfo = socket.getaddrinfo


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "psycopg: the test drives psycopg's async mode, which needs a selector event loop "
        "on Windows",
    )


def _selector_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def pytest_asyncio_loop_factories(config, item):
    """The event loop each async test runs on.

    psycopg's async mode cannot run on Windows' default loop, the Proactor:
    a test marked `psycopg` gets a selector loop there. Every other test gets
    the default loop, which the stdio MCP tests need on Windows, since only
    the Proactor runs subprocesses. On Linux, as in CI, both are the same."""
    if sys.platform == "win32" and item.get_closest_marker("psycopg"):
        return {"selector": _selector_loop}
    return {"default": asyncio.new_event_loop}


class NetworkBlocked(socket.gaierror):
    """A test resolved a host it may not reach."""


def _allowed_hosts() -> set[str]:
    hosts = {"localhost", "127.0.0.1", "::1"}
    for var in ("TEST_DATABASE_URL", "DATABASE_URL"):
        host = urlsplit(os.environ.get(var, "")).hostname
        if host:
            hosts.add(host.lower())
    return hosts


@pytest.fixture(autouse=True)
def network_guard(monkeypatch):
    """Refuse every host but the allowed ones. Yields the refused hosts; a test
    that expects a refusal clears the list."""
    allowed = _allowed_hosts()
    refused: list[str] = []

    def getaddrinfo(host, *args, **kwargs):
        name = host.decode() if isinstance(host, bytes) else host
        if not name or name.lower() in allowed:
            return _real_getaddrinfo(host, *args, **kwargs)
        refused.append(name)
        raise NetworkBlocked(socket.EAI_NONAME, f"tests may not reach {name}")

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    yield refused
    if refused:
        pytest.fail(
            f"the test tried to reach {sorted(set(refused))}: stub the provider it calls "
            "(FakeProvider and mock_llm_service_for, or mock_embedder)",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def open_api(monkeypatch):
    """Every test starts with the API open and nothing counted: the dev .env
    may set API_KEY or a low RATE_LIMIT_*. A test that needs either sets it."""
    from app.api.rate_limit import limiter
    from app.config import settings

    monkeypatch.setattr(settings, "api_key", None)
    for name in (
        "rate_limit_per_minute",
        "rate_limit_upload_per_minute",
        "rate_limit_eval_per_minute",
    ):
        monkeypatch.setattr(settings, name, 0)
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture(autouse=True)
def no_default_collection(monkeypatch):
    """No collection for WhatsApp to fall back on, as in CI: the dev .env may
    set DEFAULT_COLLECTION_ID, and the WhatsApp tests then started threads
    on a collection their database does not have. A test that needs one
    sets it."""
    from app.config import settings

    monkeypatch.setattr(settings, "default_collection_id", None)


_span_exporter = None


@pytest.fixture
def spans():
    """The spans that ended during the test, in the order they ended
    (`spans.get_finished_spans()`). One in-memory exporter for the session,
    added to the provider `configure_tracing` installs."""
    global _span_exporter
    if _span_exporter is None:
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from app.telemetry import configure_tracing

        _span_exporter = InMemorySpanExporter()
        configure_tracing().add_span_processor(SimpleSpanProcessor(_span_exporter))
    _span_exporter.clear()
    yield _span_exporter
    _span_exporter.clear()


@pytest.fixture(autouse=True)
def fresh_health_check(monkeypatch):
    """No embedder check kept from another test (app/api/v1/health.py)."""
    from app.api.v1 import health

    monkeypatch.setattr(health, "_embedder_check", None)


@pytest.fixture(autouse=True)
def fresh_model_status():
    """Each test starts with no call recorded (app/services/model_status.py)."""
    from app.services.model_status import model_status

    model_status.reset()
    yield
    model_status.reset()
