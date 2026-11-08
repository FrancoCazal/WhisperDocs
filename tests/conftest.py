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

import os
import socket
from urllib.parse import urlsplit

import pytest

# Without credentials in the environment (as in CI), botocore asks the EC2
# metadata endpoint for them when a client is created.
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")

_real_getaddrinfo = socket.getaddrinfo


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
