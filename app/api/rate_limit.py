"""Requests a minute per client on the routes that spend (audit R9).

A sliding window: a client may make `limit` requests in any 60 seconds of a
bucket; past that it gets a 429 with `Retry-After`. The client is its API key
(hashed) when the API checks keys, its IP otherwise. Each bucket has its own setting
(0 turns it off):

  questions    /query, /retrieve, /agent/chat, /agent/chat/stream,
               /health?deep=true
               RATE_LIMIT_PER_MINUTE
  uploads      the three upload routes          RATE_LIMIT_UPLOAD_PER_MINUTE
  eval         /eval/run                        RATE_LIMIT_EVAL_PER_MINUTE

The counts live in memory, in this process: the API runs as one uvicorn
process, and a restart starts them over.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Security, status

from app.api.auth import api_key_header, configured_key
from app.api.errors import APIError
from app.config import settings

WINDOW_S = 60.0
# Every this many requests, drop the clients idle for a whole window.
_SWEEP_EVERY = 1000


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = {}
        self._calls = 0

    def hit(self, bucket: str, client: str, limit: int) -> float | None:
        """Count a request. Over the limit, count nothing and return the
        seconds until the oldest request in the window leaves it."""
        now = self._clock()
        self._calls += 1
        if self._calls % _SWEEP_EVERY == 0:
            self._sweep(now)
        hits = self._hits.setdefault((bucket, client), deque())
        while hits and now - hits[0] >= WINDOW_S:
            hits.popleft()
        if len(hits) >= limit:
            return WINDOW_S - (now - hits[0])
        hits.append(now)
        return None

    def _sweep(self, now: float) -> None:
        idle = [k for k, hits in self._hits.items() if not hits or now - hits[-1] >= WINDOW_S]
        for k in idle:
            del self._hits[k]

    def reset(self) -> None:
        self._hits.clear()


limiter = RateLimiter()


def _client(request: Request, key: str | None) -> str:
    # The key names the client only when the API checks it, and by then it
    # was checked: with auth off, any made-up key per request would get a
    # fresh window.
    if key and configured_key() is not None:
        return "key:" + hashlib.sha256(key.encode()).hexdigest()[:16]
    return "ip:" + (request.client.host if request.client else "unknown")


def check_rate(request: Request, key: str | None, bucket: str, limit: int) -> None:
    if limit <= 0:
        return
    wait = limiter.hit(bucket, _client(request, key), limit)
    if wait is not None:
        seconds = max(1, math.ceil(wait))
        raise APIError(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate_limited",
            f"Too many requests: at most {limit} a minute here. Retry in {seconds} s.",
            headers={"Retry-After": str(seconds)},
        )


def rate_limit(bucket: str, per_minute: Callable[[], int]) -> Callable[..., Awaitable[None]]:
    async def dependency(request: Request, key: str | None = Security(api_key_header)) -> None:
        check_rate(request, key, bucket, per_minute())

    return dependency


questions_limit = rate_limit("questions", lambda: settings.rate_limit_per_minute)
uploads_limit = rate_limit("uploads", lambda: settings.rate_limit_upload_per_minute)
eval_limit = rate_limit("eval", lambda: settings.rate_limit_eval_per_minute)
