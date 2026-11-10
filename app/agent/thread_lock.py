"""One turn at a time per thread (audit A8).

Two turns on one thread would both start from the same checkpoint: each
would answer without the other's question, and whichever saved last would
be the thread's history. A turn waits for the one running on its thread;
`asyncio.Lock` wakes the waiters in the order they came.

The locks live in this process, which is enough for the one uvicorn worker
compose runs. Several workers would need a lock they share, such as a
Postgres advisory lock per thread.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# A lock per thread with a turn running or waiting, dropped with the last one.
_locks: dict[str, asyncio.Lock] = {}
_turns: Counter[str] = Counter()


@asynccontextmanager
async def one_turn_at_a_time(thread_id: str) -> AsyncIterator[None]:
    lock = _locks.setdefault(thread_id, asyncio.Lock())
    _turns[thread_id] += 1
    try:
        async with lock:
            yield
    finally:
        _turns[thread_id] -= 1
        if not _turns[thread_id]:
            del _turns[thread_id]
            del _locks[thread_id]


def threads_with_turns() -> set[str]:
    """The threads with a turn running or waiting."""
    return set(_locks)
