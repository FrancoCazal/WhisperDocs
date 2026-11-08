"""The summary keeps where its facts came from, and a failed one is not
retried on every turn (audit A17).

- Each user message carries what its turn searched (`asked`), and the
  summary's transcript names it on every user line.
- A summary missing a section (cut at max_tokens) or empty is not kept.
- After a failure, `check_compaction` waits `compaction_retry_messages`
  messages before compacting again.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes import summarize as summarize_module
from app.agent.nodes.check_compaction import check_compaction_node
from app.agent.nodes.summarize import _scope_names, summarize_node
from app.agent.prompts import SUMMARY_SECTIONS
from app.agent.utils.messages import SCOPE, asked
from app.config import settings
from app.services.llm.base import LLMResponse


@pytest.fixture(autouse=True)
def small_window(monkeypatch):
    monkeypatch.setattr(settings.agent, "k_recent", 2)
    monkeypatch.setattr(settings.agent, "compaction_count_threshold", 3)
    monkeypatch.setattr(settings.agent, "compaction_retry_messages", 6)


def _reply(text: str) -> LLMResponse:
    return LLMResponse(text=text, tokens_input=10, tokens_output=5, model="fake", latency_ms=1)


def _turns(n: int) -> list:
    """n turns of plain messages, then the new question."""
    messages: list = []
    for i in range(n):
        messages += [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]
    return [*messages, HumanMessage(content=f"q{n}")]


def test_a_users_message_carries_what_its_turn_searched():
    message = asked("¿Qué dice?", collection_id=3, document_id=None)

    assert message.content == "¿Qué dice?"
    assert message.additional_kwargs[SCOPE] == {"collection_id": 3, "document_id": None}
    assert datetime.fromisoformat(message.additional_kwargs["timestamp"]).tzinfo is not None


async def test_each_user_line_names_what_its_turn_searched(
    fake_provider, mock_llm_service_for, complete_summary, monkeypatch
):
    async def names(_messages, sessions=None):
        return {("collection", 3): "Contratos", ("document", 7): "Manual.pdf"}

    monkeypatch.setattr(summarize_module, "_scope_names", names)
    fake_provider.generate_response = _reply(complete_summary)
    mock_llm_service_for("app.agent.nodes.summarize", fake_provider)
    messages = [
        HumanMessage(content="before scopes"),
        AIMessage(content="a0"),
        asked("in a collection", collection_id=3),
        AIMessage(content="a1"),
        asked("in a document", collection_id=3, document_id=7),
        AIMessage(content="a2"),
        asked("everywhere", collection_id=None),
        AIMessage(content="a3"),
        asked("gone", collection_id=9),
        AIMessage(content="a4"),
        asked("the new question", collection_id=3),
    ]

    await summarize_node({"messages": messages, "summary_covered_count": 0})

    sent = fake_provider.calls[-1]["kwargs"]
    content = sent["messages"][0]["content"]
    assert "User: before scopes" in content
    assert 'User [collection "Contratos"]: in a collection' in content
    assert 'User [document "Manual.pdf"]: in a document' in content
    assert "User [all collections]: everywhere" in content
    # A collection the lookup did not name (deleted since) goes by its id.
    assert "User [collection 9]: gone" in content
    assert "Assistant: a1" in content
    assert sent["max_tokens"] == summarize_module.SUMMARY_MAX_TOKENS


@pytest.mark.parametrize("text", ["", "## User goal\nX\n\n## Decisions\n(none)"])
async def test_an_incomplete_summary_is_not_kept(fake_provider, mock_llm_service_for, text):
    fake_provider.generate_response = _reply(text)
    mock_llm_service_for("app.agent.nodes.summarize", fake_provider)
    messages = _turns(3)

    result = await summarize_node({"messages": messages, "summary": "earlier"})

    assert "summary" not in result
    assert "summary_covered_count" not in result
    assert result["summary_retry_at"] == len(messages) + 6
    assert result["metadata"]["turn_errors"] == [
        {"step": "summarize", "kind": "bad_response", "model": "fake-model"}
    ]


async def test_a_kept_summary_has_every_section_in_order(
    fake_provider, mock_llm_service_for, complete_summary
):
    shuffled = complete_summary.replace(SUMMARY_SECTIONS[0], "## Goal")
    fake_provider.generate_response = _reply(shuffled + "\n\n" + SUMMARY_SECTIONS[0])
    mock_llm_service_for("app.agent.nodes.summarize", fake_provider)

    result = await summarize_node({"messages": _turns(3)})

    assert "summary" not in result


async def test_compaction_waits_after_a_failed_summary():
    messages = _turns(3)  # 7 messages, over the threshold of 3

    waiting = await check_compaction_node({"messages": messages, "summary_retry_at": 8})
    due = await check_compaction_node({"messages": messages, "summary_retry_at": 7})
    never_failed = await check_compaction_node({"messages": messages, "summary_retry_at": None})

    assert waiting["pending_action"] is None
    assert due["pending_action"] == "summarize"
    assert never_failed["pending_action"] == "summarize"


async def test_a_kept_summary_ends_the_wait(fake_provider, mock_llm_service_for, complete_summary):
    fake_provider.generate_response = _reply(complete_summary)
    mock_llm_service_for("app.agent.nodes.summarize", fake_provider)

    result = await summarize_node({"messages": _turns(3), "summary_retry_at": 7})

    assert result["summary"] == complete_summary
    assert result["summary_retry_at"] is None


async def test_scope_names_needs_no_lookup_without_scopes():
    opened: list[bool] = []

    @asynccontextmanager
    async def sessions():
        opened.append(True)
        yield None

    assert await _scope_names(_turns(2), sessions=sessions) == {}
    assert opened == []


async def test_a_failed_lookup_leaves_the_ids():
    @asynccontextmanager
    async def broken():
        raise ConnectionError("database down")
        yield

    assert await _scope_names([asked("q", collection_id=3)], sessions=broken) == {}
