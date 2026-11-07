"""Failed answers stay in the thread but out of every prompt (audit N4)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

import app.agent.nodes.search_docs as search_docs_module
from app.agent.nodes.answer_direct import answer_direct_node
from app.agent.nodes.ask_clarification import ask_clarification_node
from app.agent.nodes.search_docs import search_docs_node
from app.agent.utils.fixed_replies import fixed_reply
from app.agent.utils.messages import (
    FAILED_TURN,
    build_prompt_messages,
    failed_answer,
    format_messages,
)


def _boom(**_):
    raise RuntimeError("provider down")


def test_format_messages_drops_a_failed_answer_with_its_question():
    messages = [
        HumanMessage(content="q1"),
        AIMessage(content="a1"),
        HumanMessage(content="q2"),
        failed_answer("I had a problem…"),
        HumanMessage(content="q2 again"),
    ]

    assert format_messages(messages) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2 again"},
    ]


def test_a_failed_answer_whose_question_was_compacted_goes_alone():
    # The tail after compaction can start right at the failed answer.
    messages = [failed_answer("I had a problem…"), HumanMessage(content="q3")]

    assert format_messages(messages) == [{"role": "user", "content": "q3"}]


async def test_a_generation_failure_is_marked_and_left_out_of_the_next_prompt(
    fake_provider,
    mock_llm_service_for,
    mock_embedder,
    mock_retriever,
    mock_session_local,
    rag_chunk,
    monkeypatch,
):
    mock_retriever([rag_chunk(0)])
    fake_provider.generate_stream = _boom  # type: ignore[assignment]
    fake_provider.generate_structured = _boom  # type: ignore[assignment]
    mock_llm_service_for("app.agent.nodes.search_docs", fake_provider)
    question = HumanMessage(content="How is the chunker configured?")

    result = await search_docs_node({"messages": [question], "collection_id": None})

    answer = result["messages"][0]
    assert answer.content == fixed_reply("generation_failed", [question])
    assert answer.additional_kwargs[FAILED_TURN] is True
    assert result["metadata"]["turn_failed"] is True

    next_turn = {"messages": [question, answer, HumanMessage(content="Try again?")]}
    assert build_prompt_messages(next_turn) == [{"role": "user", "content": "Try again?"}]


async def test_an_answer_is_not_marked(
    fake_provider,
    mock_llm_service_for,
    mock_embedder,
    mock_retriever,
    mock_session_local,
    rag_chunk,
    monkeypatch,
):
    mock_retriever([rag_chunk(0)])
    mock_llm_service_for("app.agent.nodes.search_docs", fake_provider)
    monkeypatch.setattr(search_docs_module.settings.agent, "rag_structured_output", False)

    result = await search_docs_node(
        {"messages": [HumanMessage(content="¿quién es tino?")], "collection_id": None}
    )

    assert FAILED_TURN not in result["messages"][0].additional_kwargs
    assert result["metadata"]["turn_failed"] is False


async def test_a_direct_answer_failure_is_marked(fake_provider, mock_llm_service_for):
    fake_provider.generate_stream = _boom  # type: ignore[assignment]
    mock_llm_service_for("app.agent.nodes.answer_direct", fake_provider)

    result = await answer_direct_node({"messages": [HumanMessage(content="hola")]})

    assert result["messages"][0].additional_kwargs[FAILED_TURN] is True
    assert result["metadata"]["turn_failed"] is True


async def test_a_clarification_failure_still_asks_and_is_not_marked(
    fake_provider, mock_llm_service_for
):
    fake_provider.generate_stream = _boom  # type: ignore[assignment]
    mock_llm_service_for("app.agent.nodes.ask_clarification", fake_provider)

    result = await ask_clarification_node({"messages": [HumanMessage(content="help")]})

    message = result["messages"][0]
    assert message.content == "Can you give me more details about what you need?"
    assert FAILED_TURN not in message.additional_kwargs
