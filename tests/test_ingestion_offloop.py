"""Ingestion keeps the event loop free while it parses (audit R4).

pdfplumber, the tiktoken chunker and langdetect are CPU-bound. Called inline
in an `async def`, a large PDF froze every other request, chat SSE included,
until it finished. The scan for instruction-like text (audit M-5) is
CPU-bound too, about a millisecond a chunk.

Each test wraps such a step in `OffTheLoop`: the step runs only once the
event loop has run a callback the step scheduled. In a worker thread, the
loop runs it at once. On the loop, the loop is the thread that waits, so the
callback cannot run and the step fails. No clock is read: these tests once
measured the loop's longest gap against 150 ms, and a collection of the
GC's oldest generation in the full suite (up to 1.2 s) failed them about
one run in four, with the steps off the loop all along (BE-26).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

import pytest

from app.models import Document, IngestionStatus
from app.services import ingestion

# How long a step waits for the loop before it fails. Off the loop the
# callback runs within milliseconds; this is spent only when a step is on it.
WAIT_S = 5.0
PARAGRAPH = (
    "The support team answers customer tickets within one business day. "
    "Every new hire shadows a senior engineer during the first week, "
    "and the on-call rotation starts only after the second month. "
)
TEXT = "# Onboarding guide\n\n" + PARAGRAPH * 30


class OffTheLoop:
    """A CPU-bound step that runs only if the event loop runs meanwhile."""

    def __init__(self, step: Callable) -> None:
        self.step = step
        self.loop = asyncio.get_running_loop()
        self.calls = 0

    def __call__(self, *args):
        answered = threading.Event()
        self.loop.call_soon_threadsafe(answered.set)
        if not answered.wait(WAIT_S):
            raise AssertionError(f"{self.step.__name__} ran on the event loop")
        self.calls += 1
        return self.step(*args)


def _extract(_file_bytes: bytes, _file_type: str) -> str:
    return TEXT


class _Result:
    def all(self):
        return []


class _FakeSession:
    def __init__(self) -> None:
        self.added: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def execute(self, *_a, **_k):
        return _Result()

    def add(self, obj) -> None:
        self.added.append(obj)

    def add_all(self, objs) -> None:
        self.added.extend(objs)

    async def flush(self) -> None:
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = 1

    async def commit(self) -> None:
        pass

    async def refresh(self, _obj) -> None:
        pass


class _FakeEmbedder:
    model_id = "fake-embedding"

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1536 for _ in texts]


@pytest.fixture
async def steps(monkeypatch) -> dict[str, OffTheLoop]:
    """Parsing, chunking and the scan, each checked to run off the loop."""
    wrapped = {
        "extract": OffTheLoop(_extract),
        "chunk": OffTheLoop(ingestion._chunk_and_detect),
        "scan": OffTheLoop(ingestion._scan),
    }
    monkeypatch.setattr(ingestion, "extract_text", wrapped["extract"])
    monkeypatch.setattr(ingestion, "_chunk_and_detect", wrapped["chunk"])
    monkeypatch.setattr(ingestion, "_scan", wrapped["scan"])
    return wrapped


async def test_the_background_worker_parses_off_the_loop(steps, monkeypatch):
    statuses: list[IngestionStatus] = []

    async def record_status(_job_id, **kwargs):
        if kwargs.get("status") is not None:
            statuses.append(kwargs["status"])

    monkeypatch.setattr(ingestion, "_set_status", record_status)
    monkeypatch.setattr(ingestion, "SessionLocal", _FakeSession)
    monkeypatch.setattr(ingestion, "EmbeddingService", _FakeEmbedder)

    await ingestion.run_ingestion(
        job_id=1, file_bytes=b"%PDF", filename="doc.md", title=None, collection_id=1
    )

    assert statuses[-1] == IngestionStatus.COMPLETE
    assert {name: step.calls for name, step in steps.items()} == {
        "extract": 1,
        "chunk": 1,
        "scan": 1,
    }


async def test_the_instruction_scan_runs_off_the_loop(steps):
    document = Document(id=1, filename="doc.md", language="en")

    rows = await ingestion.chunk_rows(
        document, [ingestion.FreshChunk(0, PARAGRAPH, 40, "h")], [[0.0]], "fake-embedding"
    )

    assert len(rows) == 1
    assert steps["scan"].calls == 1


async def test_a_sync_upload_parses_off_the_loop(steps):
    from app.api.v1.documents import _ingest_single

    class _Upload:
        filename = "doc.md"

        async def read(self, _size: int = -1) -> bytes:
            return b"# Title"

    document = await _ingest_single(_Upload(), 1, None, _FakeSession(), _FakeEmbedder())

    assert document.chunk_count > 0
    assert document.language == "en"
    assert {name: step.calls for name, step in steps.items()} == {
        "extract": 1,
        "chunk": 1,
        "scan": 1,
    }


async def test_a_step_on_the_loop_fails(monkeypatch):
    """The check itself: a step called inline cannot pass."""
    monkeypatch.setitem(globals(), "WAIT_S", 0.2)
    step = OffTheLoop(_extract)

    with pytest.raises(AssertionError, match="_extract ran on the event loop"):
        step(b"", "md")
    await asyncio.to_thread(step, b"", "md")

    assert step.calls == 1
