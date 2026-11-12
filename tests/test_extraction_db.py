"""`POST /documents/{id}/extract` (audit M-8) against a real Postgres: the
passages come from the document's chunks, in order, with their sections and
pages, and each citation names its chunk. The model is a fake.

Needs a database migrated to head (see test_retrieval_db.py); skipped when
TEST_DATABASE_URL is not set. The test runs inside a transaction that is
rolled back.
"""

from __future__ import annotations

import os

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.db import get_db
from app.main import app
from app.models import Chunk, Collection, Document
from app.services import extraction
from app.services.embedder import get_embedder
from app.services.llm.base import LLMResponse

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")

SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total": {"type": "number"},
    },
    "required": ["invoice_number", "total"],
}


@pytest.fixture
async def session():
    engine = create_async_engine(DB_URL)
    async with engine.connect() as conn:
        outer = await conn.begin()
        async with AsyncSession(
            bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
        ) as s:
            yield s
        await outer.rollback()
    await engine.dispose()


class _Provider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        model = kwargs["response_model"]
        answer = model.model_validate(
            {
                **dict.fromkeys(model.model_fields),
                "invoice_number": "INV-2041",
                "invoice_number_quote": "Invoice INV-2041",
                "invoice_number_passage": 1,
                "total": 1250.0,
                "total_quote": "Total due: 1,250.00",
                "total_passage": 2,
            }
        )
        return answer, LLMResponse(
            text="", tokens_input=500, tokens_output=80, model="fake-model", latency_ms=3
        )


async def test_the_fields_cite_the_chunks_that_state_them(session, monkeypatch):
    collection = Collection(name="extract-db", description="")
    session.add(collection)
    await session.flush()
    document = Document(
        collection_id=collection.id,
        title="invoice.pdf",
        filename="invoice.pdf",
        file_type="pdf",
        raw_text="Invoice INV-2041\fTotal due: 1,250.00",
        chunk_count=2,
        total_tokens=12,
    )
    session.add(document)
    await session.flush()
    # Stored out of order: the passages follow chunk_index.
    for index, text, page in ((1, "Total due: 1,250.00", 2), (0, "Invoice INV-2041", 1)):
        session.add(
            Chunk(
                document_id=document.id,
                chunk_index=index,
                text=text,
                token_count=6,
                embedding=[0.0] * 1535 + [1.0],
                section="Billing",
                page_start=page,
                page_end=page,
            )
        )
    await session.flush()
    provider = _Provider()
    monkeypatch.setattr(extraction, "llm_service_for", lambda _role: (provider, "fake", 0.0))

    async def db():
        yield session

    app.dependency_overrides[get_db] = db
    app.dependency_overrides[get_embedder] = lambda: None
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
            r = await c.post(f"/documents/{document.id}/extract", json={"json_schema": SCHEMA})
            missing = await c.post("/documents/2000000000/extract", json={"json_schema": SCHEMA})
    finally:
        app.dependency_overrides.clear()

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["values"] == {"invoice_number": "INV-2041", "total": 1250.0}
    total = body["fields"]["total"]["citation"]
    assert (total["chunk_index"], total["section"], total["page_start"]) == (1, "Billing", 2)
    assert total["verified"] is True
    assert body["metadata"]["passages_used"] == body["metadata"]["passages_total"] == 2
    (prompt,) = provider.prompts
    assert prompt.index("[1] (chunk 0") < prompt.index("[2] (chunk 1")
    assert missing.status_code == 404
