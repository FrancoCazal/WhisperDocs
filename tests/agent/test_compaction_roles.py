"""Role order after compaction (audit A2).

Bedrock Converse rejects a conversation whose first message is the
assistant's. Compaction runs when the last message is the user's, so the
message count is odd; with k_recent = 8 the old cut point `len - 8` landed on an
AIMessage and every later RAG turn in that thread failed.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes.summarize import summarize_node
from app.agent.utils.messages import build_prompt_messages, drop_leading_assistant
from app.agent.utils.query_rewrite import rewrite_query_for_retrieval
from app.services.llm.base import LLMResponse


@pytest.fixture(autouse=True)
def summarizes(fake_provider, complete_summary):
    """The fake returns a summary `summarize` keeps."""
    fake_provider.generate_response = LLMResponse(
        text=complete_summary, tokens_input=10, tokens_output=5, model="fake", latency_ms=1
    )


def _alternating(n_turns: int) -> list:
    """n_turns full turns (user + assistant) followed by the new user message."""
    messages: list = []
    for i in range(n_turns):
        messages.append(HumanMessage(content=f"q{i}"))
        messages.append(AIMessage(content=f"a{i}"))
    messages.append(HumanMessage(content=f"q{n_turns}"))
    return messages


async def test_summarize_cut_point_leaves_a_user_message_first(fake_provider, mock_llm_service_for):
    mock_llm_service_for("app.agent.nodes.summarize", fake_provider)
    messages = _alternating(15)  # 31 messages: the count threshold (> 30) fires here

    out = await summarize_node({"messages": messages, "summary_covered_count": 0})

    covered = out["summary_covered_count"]
    assert isinstance(messages[covered], HumanMessage)
    # Never more than k_recent (8) messages left uncompacted.
    assert len(messages) - covered <= 8


async def test_summarize_second_compaction_stays_aligned(fake_provider, mock_llm_service_for):
    mock_llm_service_for("app.agent.nodes.summarize", fake_provider)
    messages = _alternating(30)  # 61 messages

    out = await summarize_node(
        {"messages": messages, "summary_covered_count": 24, "summary": "earlier"}
    )

    assert isinstance(messages[out["summary_covered_count"]], HumanMessage)


async def test_summarize_sends_the_window_as_one_user_message(fake_provider, mock_llm_service_for):
    """The window ends with an assistant message. Replayed as turns, Claude on
    Bedrock reads it as a prefill and returns an empty summary (seen live)."""
    mock_llm_service_for("app.agent.nodes.summarize", fake_provider)
    messages = _alternating(15)

    await summarize_node({"messages": messages, "summary_covered_count": 0, "summary": "s0"})

    sent = fake_provider.calls[-1]["kwargs"]["messages"]
    assert [m["role"] for m in sent] == ["user"]
    content = sent[0]["content"]
    assert content.startswith("Previous summary (absorb into the new one):\ns0")
    assert "User: q0\n\nAssistant: a0" in content
    assert content.rstrip().endswith("required structure.")


async def test_prompt_after_compaction_starts_with_user():
    messages = _alternating(15)
    state = {"messages": messages, "summary_covered_count": 24, "summary": "s"}

    prompt = build_prompt_messages(state)

    assert prompt[0]["role"] == "system"  # the summary
    assert prompt[1]["role"] == "user"


def test_prompt_heals_threads_compacted_before_the_fix():
    """A checkpoint saved with an odd `summary_covered_count` still yields a
    prompt that opens with a user message."""
    messages = _alternating(15)
    state = {"messages": messages, "summary_covered_count": 23, "summary": "s"}
    assert isinstance(messages[23], AIMessage)

    prompt = build_prompt_messages(state)
    rag_history = build_prompt_messages(state, exclude_last=True)

    assert [m["role"] for m in prompt[:2]] == ["system", "user"]
    assert [m["role"] for m in rag_history[:2]] == ["system", "user"]
    assert rag_history[-1]["role"] == "assistant"  # search_docs appends the user prompt


def test_drop_leading_assistant_keeps_the_rest():
    formatted = [
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "b"},
    ]
    assert drop_leading_assistant(formatted) == formatted[1:]
    assert drop_leading_assistant([]) == []


async def test_query_rewrite_history_starts_with_user(fake_provider, mock_llm_service_for):
    mock_llm_service_for("app.agent.utils.query_rewrite", fake_provider)
    history = _alternating(15)[23:]  # odd cut: starts with an AIMessage

    await rewrite_query_for_retrieval(history=history, current="and that?", summary="s")

    sent = fake_provider.calls[-1]["kwargs"]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[1]["role"] == "user"
