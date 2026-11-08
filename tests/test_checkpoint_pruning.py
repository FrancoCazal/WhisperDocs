"""The checkpoint sweep (audit N6): each thread keeps its latest checkpoint.

A thread another transaction changes mid-prune is left for the next sweep,
one that fails does not stop the others, and a failed sweep does not stop the
loop. Against Postgres: test_checkpoint_pruning_db.py.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from psycopg.errors import DeadlockDetected, SerializationFailure

from app.agent import checkpointer
from app.agent.checkpointer import Pruned, prune_checkpoints, prune_threads
from app.config import settings


class _Saver:
    def __init__(self, raises: dict[str, BaseException] | None = None):
        self.raises = raises or {}
        self.pruned: list[tuple[list[str], str]] = []

    async def aprune(self, thread_ids, *, strategy):
        self.pruned.append((list(thread_ids), strategy))
        if thread_ids[0] in self.raises:
            raise self.raises[thread_ids[0]]


@pytest.fixture(autouse=True)
async def no_sweep_left():
    yield
    await checkpointer.stop_pruning()


async def test_each_thread_keeps_its_latest_checkpoint_in_its_own_prune():
    saver = _Saver()

    result = await prune_threads(saver, ["a", "b"])

    assert saver.pruned == [(["a"], "keep_latest"), (["b"], "keep_latest")]
    assert result == Pruned(threads=2)


async def test_a_busy_or_failing_thread_leaves_the_others_to_go_ahead(caplog):
    saver = _Saver(
        {
            "turn": SerializationFailure("could not serialize access"),
            "delete": DeadlockDetected("deadlock detected"),
            "broken": RuntimeError("boom"),
        }
    )

    with caplog.at_level(logging.WARNING, logger="app.agent.checkpointer"):
        result = await prune_threads(saver, ["turn", "delete", "broken", "fine"])

    assert [ids[0] for ids, _ in saver.pruned] == ["turn", "delete", "broken", "fine"]
    assert result == Pruned(threads=1, busy=2, failed=1)
    # A busy thread is routine; only the failure is logged.
    assert [r.getMessage() for r in caplog.records] == ["checkpoint_prune_failed thread_id=broken"]


async def test_a_sweep_prunes_the_threads_with_more_than_their_latest(monkeypatch, caplog):
    saver = _Saver()

    async def bloated(_saver):
        return ["a", "b"]

    monkeypatch.setattr(checkpointer, "threads_to_prune", bloated)
    with caplog.at_level(logging.INFO, logger="app.agent.checkpointer"):
        result = await prune_checkpoints(saver)

    assert result == Pruned(threads=2)
    assert "checkpoints_pruned threads=2 busy=0 failed=0" in caplog.text


async def test_a_sweep_with_nothing_to_prune_logs_nothing(monkeypatch, caplog):
    async def none(_saver):
        return []

    monkeypatch.setattr(checkpointer, "threads_to_prune", none)
    with caplog.at_level(logging.INFO, logger="app.agent.checkpointer"):
        assert await prune_checkpoints(_Saver()) == Pruned()

    assert caplog.records == []


async def test_the_sweep_repeats_after_a_failed_one(monkeypatch):
    monkeypatch.setattr(settings.agent, "checkpoint_prune_interval_s", 0.01)
    sweeps: list[object] = []
    second = asyncio.Event()

    async def sweep(saver):
        sweeps.append(saver)
        if len(sweeps) == 1:
            raise ConnectionError("database down")
        second.set()
        return Pruned()

    monkeypatch.setattr(checkpointer, "prune_checkpoints", sweep)
    saver = _Saver()

    checkpointer.start_pruning(saver)
    checkpointer.start_pruning(saver)  # a second start keeps the one sweep
    assert [t.get_name() for t in asyncio.all_tasks()].count("checkpoint-pruning") == 1
    await asyncio.wait_for(second.wait(), timeout=2)
    task = checkpointer._pruning
    await checkpointer.stop_pruning()

    assert sweeps[:2] == [saver, saver]
    assert task is not None and task.cancelled()
    assert checkpointer._pruning is None


async def test_the_sweep_waits_its_interval_before_the_first_prune(monkeypatch):
    monkeypatch.setattr(settings.agent, "checkpoint_prune_interval_s", 60.0)
    sweeps: list[object] = []

    async def sweep(saver):
        sweeps.append(saver)
        return Pruned()

    monkeypatch.setattr(checkpointer, "prune_checkpoints", sweep)

    checkpointer.start_pruning(_Saver())
    await asyncio.sleep(0.05)

    assert sweeps == []


async def test_an_interval_of_zero_turns_pruning_off(monkeypatch):
    monkeypatch.setattr(settings.agent, "checkpoint_prune_interval_s", 0)

    checkpointer.start_pruning(_Saver())

    assert checkpointer._pruning is None
    await checkpointer.stop_pruning()  # nothing to stop
