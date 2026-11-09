"""Ingestion keeps the event loop free while it parses (audit R4).

pdfplumber, the tiktoken chunker and langdetect are CPU-bound. Called inline
in an `async def`, a large PDF froze every other request, chat SSE included,
until it finished. A fake extractor that blocks for 0.3 s stands in for the
parse; a ticker coroutine measures the longest gap the loop takes to run it.
The scan for instruction-like text (audit M-5) is CPU-bound too, about a
millisecond a chunk, and a fake scan stands in for it the same way.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.models import Document, IngestionStatus
from app.services import ingestion

BLOCK_S = 0.3
PARAGRAPH = (
    "The support team answers customer tickets within one business day. "
    "Every new hire shadows a senior engineer during the first week, "
    "and the on-call rotation starts only after the second month. "
)
TEXT = "# Onboarding guide\n\n" + PARAGRAPH * 30


def _blocking_extract(_file_bytes: bytes, _file_type: str) -> str:
    time.sleep(BLOCK_S)  # stands in for pdfplumber holding the thread
    return TEXT


async def _max_loop_gap(coro) -> tuple[float, object]:
    """Longest time the loop went without running a 10 ms ticker while `coro` ran."""
    gaps: list[float] = []
    last = [time.perf_counter()]

    async def ticker() -> None:
        while True:
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last[0])
            last[0] = now

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)  # let the ticker start before the work does
    try:
        result = await coro
    finally:
        gaps.append(time.perf_counter() - last[0])  # the stretch since the last tick
        task.cancel()
    return max(gaps), result


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
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1536 for _ in texts]


@pytest.fixture(scope="module", autouse=True)
def _warm_tokenizer_and_langdetect():
    # The first call in a process builds tiktoken's BPE table and loads the
    # langdetect profiles, which holds the GIL for about 0.1 s. That happens
    # once per process; the tests measure the steady state.
    from app.services.chunker import TextChunker
    from app.services.lang import detect_language

    TextChunker()
    detect_language(TEXT)


@pytest.fixture
def blocking_extractor(monkeypatch):
    monkeypatch.setattr(ingestion, "extract_text", _blocking_extract)


async def test_background_worker_does_not_block_the_loop(blocking_extractor, monkeypatch):
    statuses: list[IngestionStatus] = []

    async def record_status(_job_id, **kwargs):
        if kwargs.get("status") is not None:
            statuses.append(kwargs["status"])

    monkeypatch.setattr(ingestion, "_set_status", record_status)
    monkeypatch.setattr(ingestion, "SessionLocal", _FakeSession)
    monkeypatch.setattr(ingestion, "EmbeddingService", _FakeEmbedder)

    gap, _ = await _max_loop_gap(
        ingestion.run_ingestion(
            job_id=1, file_bytes=b"%PDF", filename="doc.md", title=None, collection_id=1
        )
    )

    assert statuses[-1] == IngestionStatus.COMPLETE
    assert gap < BLOCK_S / 2


async def test_the_instruction_scan_does_not_block_the_loop(monkeypatch):
    def blocking_scan(_text: str) -> list[str]:
        time.sleep(BLOCK_S)  # stands in for the rules over a large document
        return []

    monkeypatch.setattr(ingestion, "instruction_flags", blocking_scan)
    document = Document(id=1, filename="doc.md", language="en")

    gap, rows = await _max_loop_gap(
        ingestion.chunk_rows(document, [(0, PARAGRAPH, 40, "h")], [[0.0]])
    )

    assert len(rows) == 1
    assert gap < BLOCK_S / 2


async def test_sync_upload_does_not_block_the_loop(blocking_extractor):
    from app.api.v1.documents import _ingest_single

    class _Upload:
        filename = "doc.md"

        async def read(self, _size: int = -1) -> bytes:
            return b"# Title"

    gap, document = await _max_loop_gap(
        _ingest_single(_Upload(), 1, None, _FakeSession(), _FakeEmbedder())
    )

    assert document.chunk_count > 0
    assert document.language == "en"
    assert gap < BLOCK_S / 2
