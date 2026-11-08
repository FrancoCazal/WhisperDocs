"""Both connection pools check a connection before handing it out, so the
requests after a Postgres restart do not draw dead connections."""

from __future__ import annotations

import app.agent.checkpointer as checkpointer
from app.db import engine


def test_the_engine_pings_a_connection_on_checkout():
    assert engine.sync_engine.pool._pre_ping is True


async def test_the_checkpointer_pool_checks_a_connection_on_checkout(monkeypatch):
    created: dict = {}

    class _Pool:
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

        @staticmethod
        async def check_connection(_conn) -> None:
            pass

        async def open(self) -> None:
            pass

    class _Saver:
        def __init__(self, conn) -> None:
            pass

        async def setup(self) -> None:
            pass

    monkeypatch.setattr(checkpointer, "AsyncConnectionPool", _Pool)
    monkeypatch.setattr(checkpointer, "AsyncPostgresSaver", _Saver)
    monkeypatch.setattr(checkpointer, "_pool", None)
    monkeypatch.setattr(checkpointer, "_checkpointer", None)

    await checkpointer.init_checkpointer()

    assert created["check"] is _Pool.check_connection
