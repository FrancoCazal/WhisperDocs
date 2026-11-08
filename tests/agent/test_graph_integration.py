"""End-to-end graph tests with MemorySaver + mocked providers.

These exercise the full StateGraph topology: router edges, reducer,
checkpointer persistence. No live LLM calls, no Postgres.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_agent_graph


@pytest.fixture
def compiled_agent():
    return build_agent_graph().compile(checkpointer=MemorySaver())


def _stub_all_nodes(
    monkeypatch,
    fake_provider,
    classifier_response,
    *,
    action: str,
    answer_text: str = "stub answer",
):
    """Wire the fake provider into every node's llm_service_for + retriever stubs."""
    from app.services.llm.base import LLMResponse

    fake_provider.structured_response = classifier_response(action)
    fake_provider.generate_response = LLMResponse(
        text=answer_text,
        tokens_input=100,
        tokens_output=30,
        model="fake",
        latency_ms=150,
    )

    for mod in (
        "app.agent.nodes.decide_action",
        "app.agent.nodes.ask_clarification",
        "app.agent.nodes.answer_direct",
        "app.agent.nodes.search_docs",
        "app.agent.nodes.summarize",
    ):
        monkeypatch.setattr(
            f"{mod}.llm_service_for", lambda _n, _p=fake_provider: (_p, "fake", 0.0)
        )
    monkeypatch.setattr(
        "app.agent.utils.query_rewrite.llm_service_for",
        lambda _n, _p=fake_provider: (_p, "fake", 0.0),
    )


# ---------- a turn cut short ----------


async def test_a_turn_cut_short_leaves_the_next_prompt_valid(
    compiled_agent,
    fake_provider,
    classifier_response,
    monkeypatch,
):
    """The client closing the stream cancels the run mid-generation: the
    question stays checkpointed with no answer, and the next turn's prompt
    must still alternate user and assistant."""
    _stub_all_nodes(monkeypatch, fake_provider, classifier_response, action="answer_direct")
    config = {"configurable": {"thread_id": "cut"}, "recursion_limit": 10}
    generating = asyncio.Event()
    answer_stream = fake_provider.generate_stream

    async def hang(**_):
        generating.set()
        await asyncio.Event().wait()
        yield  # pragma: no cover - never reached

    fake_provider.generate_stream = hang  # type: ignore[assignment]

    async def run() -> None:
        async for _ in compiled_agent.astream_events(
            {"messages": [HumanMessage(content="first, cut short")], "collection_id": None},
            config=config,
            version="v2",
        ):
            pass

    task = asyncio.create_task(run())
    await asyncio.wait_for(generating.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    cut = await compiled_agent.aget_state(config)
    assert [m.content for m in cut.values["messages"]] == ["first, cut short"]

    fake_provider.generate_stream = answer_stream  # type: ignore[assignment]
    fake_provider.calls.clear()
    await compiled_agent.ainvoke(
        {"messages": [HumanMessage(content="second")], "collection_id": None}, config=config
    )

    (answer_call,) = [c for c in fake_provider.calls if c["method"] == "generate_stream"]
    roles = [m["role"] for m in answer_call["kwargs"]["messages"]]
    assert roles == ["user"]
    assert answer_call["kwargs"]["messages"][0]["content"] == "second"


# ---------- routing smoke tests ----------


async def test_graph_routes_to_answer_direct(
    compiled_agent,
    fake_provider,
    classifier_response,
    monkeypatch,
):
    _stub_all_nodes(
        monkeypatch,
        fake_provider,
        classifier_response,
        action="answer_direct",
        answer_text="¡Hola!",
    )

    result = await compiled_agent.ainvoke(
        {"messages": [HumanMessage(content="hola")], "collection_id": None},
        config={"configurable": {"thread_id": "t1"}, "recursion_limit": 10},
    )

    assert result["pending_action"] == "answer_direct"
    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert final.content == "¡Hola!"
    assert (result.get("retrieved_chunks") or []) == []


async def test_graph_routes_to_ask_clarification(
    compiled_agent,
    fake_provider,
    classifier_response,
    monkeypatch,
):
    _stub_all_nodes(
        monkeypatch,
        fake_provider,
        classifier_response,
        action="ask_clarification",
        answer_text="¿Podés darme más contexto?",
    )

    result = await compiled_agent.ainvoke(
        {"messages": [HumanMessage(content="ayuda")], "collection_id": None},
        config={"configurable": {"thread_id": "t2"}, "recursion_limit": 10},
    )

    assert result["pending_action"] == "ask_clarification"
    assert "contexto" in result["messages"][-1].content


async def test_graph_routes_to_search_docs_with_rag(
    compiled_agent,
    fake_provider,
    classifier_response,
    mock_embedder,
    mock_retriever,
    mock_session_local,
    rag_chunk,
    monkeypatch,
):
    mock_retriever([rag_chunk(0), rag_chunk(1, rel=0.75)])
    _stub_all_nodes(
        monkeypatch,
        fake_provider,
        classifier_response,
        action="search_docs",
        answer_text="Respuesta grounded [1].",
    )

    result = await compiled_agent.ainvoke(
        {
            "messages": [HumanMessage(content="¿qué es pgvector?")],
            "collection_id": 3,
        },
        config={"configurable": {"thread_id": "t3"}, "recursion_limit": 10},
    )

    assert result["pending_action"] == "search_docs"
    assert len(result["retrieved_chunks"]) == 2
    assert result["retrieved_chunks"][0]["document_title"] == "tino_lore.md"
    assert result["metadata"]["retrieval_chunks_count"] == 2


async def test_a_classifier_outage_still_answers_from_the_documents(
    compiled_agent,
    fake_provider,
    classifier_response,
    mock_embedder,
    mock_retriever,
    mock_session_local,
    rag_chunk,
    monkeypatch,
):
    """The classifier's model refusing every call (Vertex's 403 for a GCP
    project without billing) sent each question to clarification, which
    failed the same way and asked for more details (BE-9). The answer now
    says the classifier failed, and why (BE-11)."""
    from google.genai.errors import ClientError
    from langchain_google_genai.chat_models import GooglePermissionDeniedError

    from app.config import settings

    monkeypatch.setattr(settings.agent, "rag_structured_output", False)
    mock_retriever([rag_chunk(0)])
    _stub_all_nodes(
        monkeypatch,
        fake_provider,
        classifier_response,
        action="search_docs",
        answer_text="Respuesta grounded [1].",
    )

    async def refused(**_):
        # As langchain-google-genai raises it: its own type, from the SDK's.
        cause = ClientError(
            403,
            {
                "error": {
                    "code": 403,
                    "message": "This API method requires billing to be enabled.",
                    "status": "PERMISSION_DENIED",
                }
            },
        )
        raise GooglePermissionDeniedError("Error calling model (Forbidden)") from cause

    fake_provider.generate_structured = refused  # type: ignore[assignment]

    result = await compiled_agent.ainvoke(
        {
            "messages": [HumanMessage(content="tell me about rag fundamentals")],
            "collection_id": None,
        },
        config={"configurable": {"thread_id": "outage"}, "recursion_limit": 10},
    )

    assert result["pending_action"] == "search_docs"
    assert result["messages"][-1].content == "Respuesta grounded [1]."
    assert result["metadata"]["turn_failed"] is False
    assert len(result["retrieved_chunks"]) == 1
    errors = result["messages"][-1].additional_kwargs["errors"]
    assert [(e["step"], e["kind"]) for e in errors] == [("classifier", "billing")]
    assert result["metadata"]["turn_errors"] == errors


async def test_a_turn_without_generation_reports_no_generation_tokens(
    compiled_agent,
    fake_provider,
    classifier_response,
    monkeypatch,
):
    """Out of scope calls only the classifier: its tokens are not the answer's."""
    _stub_all_nodes(monkeypatch, fake_provider, classifier_response, action="out_of_scope")
    cfg = {"configurable": {"thread_id": "oos-tokens"}, "recursion_limit": 10}

    result = await compiled_agent.ainvoke(
        {"messages": [HumanMessage(content="What is the weather in Madrid?")]}, config=cfg
    )

    md = result["metadata"]
    assert md["generation_model"] is None
    assert md.get("generation_tokens_input", 0) == 0
    assert md["classifier_tokens_input"] == 50


# ---------- multi-turn memory ----------


async def test_multi_turn_same_thread_accumulates_history(
    compiled_agent,
    fake_provider,
    classifier_response,
    monkeypatch,
):
    _stub_all_nodes(
        monkeypatch,
        fake_provider,
        classifier_response,
        action="answer_direct",
        answer_text="ack",
    )

    cfg = {"configurable": {"thread_id": "multi-1"}, "recursion_limit": 10}

    r1 = await compiled_agent.ainvoke(
        {"messages": [HumanMessage(content="turn 1")], "collection_id": None}, config=cfg
    )
    assert len(r1["messages"]) == 2  # human + ai

    r2 = await compiled_agent.ainvoke(
        {"messages": [HumanMessage(content="turn 2")], "collection_id": None}, config=cfg
    )
    # Checkpointer reloaded prior state → prior 2 + new 2
    assert len(r2["messages"]) == 4
    assert r2["messages"][0].content == "turn 1"
    assert r2["messages"][-2].content == "turn 2"


async def test_thread_isolation(
    compiled_agent,
    fake_provider,
    classifier_response,
    monkeypatch,
):
    _stub_all_nodes(monkeypatch, fake_provider, classifier_response, action="answer_direct")

    cfg_a = {"configurable": {"thread_id": "thread-a"}, "recursion_limit": 10}
    cfg_b = {"configurable": {"thread_id": "thread-b"}, "recursion_limit": 10}

    await compiled_agent.ainvoke(
        {"messages": [HumanMessage(content="thread A only")]}, config=cfg_a
    )
    r_b = await compiled_agent.ainvoke(
        {"messages": [HumanMessage(content="thread B only")]}, config=cfg_b
    )

    # Thread B sees only its own message + response
    assert len(r_b["messages"]) == 2
    assert r_b["messages"][0].content == "thread B only"


# ---------- transient field reset across turns ----------


async def test_transient_fields_reset_between_turns(
    compiled_agent,
    fake_provider,
    classifier_response,
    mock_embedder,
    mock_retriever,
    mock_session_local,
    rag_chunk,
    monkeypatch,
):
    """Turn 1: RAG populates retrieved_chunks.
    Turn 2: answer_direct — must NOT carry over chunks from turn 1.
    """
    from app.services.llm.base import LLMResponse

    cfg = {"configurable": {"thread_id": "reset-test"}, "recursion_limit": 10}

    # Turn 1 — search_docs path
    mock_retriever([rag_chunk(0)])
    fake_provider.structured_response = classifier_response("search_docs")
    fake_provider.generate_response = LLMResponse(
        text="grounded [1]", tokens_input=100, tokens_output=20, model="fake", latency_ms=100
    )

    for mod in (
        "app.agent.nodes.decide_action",
        "app.agent.nodes.ask_clarification",
        "app.agent.nodes.answer_direct",
        "app.agent.nodes.search_docs",
        "app.agent.nodes.summarize",
    ):
        monkeypatch.setattr(
            f"{mod}.llm_service_for", lambda _n, _p=fake_provider: (_p, "fake", 0.0)
        )
    monkeypatch.setattr(
        "app.agent.utils.query_rewrite.llm_service_for",
        lambda _n, _p=fake_provider: (fake_provider, "fake", 0.0),
    )

    r1 = await compiled_agent.ainvoke(
        {"messages": [HumanMessage(content="q1")], "collection_id": 1}, config=cfg
    )
    assert len(r1["retrieved_chunks"]) == 1

    # Turn 2 — switch classifier to answer_direct. No retrieval this turn.
    fake_provider.structured_response = classifier_response("answer_direct")
    fake_provider.generate_response = LLMResponse(
        text="thanks!", tokens_input=5, tokens_output=3, model="fake", latency_ms=50
    )

    r2 = await compiled_agent.ainvoke(
        {"messages": [HumanMessage(content="gracias")], "collection_id": 1}, config=cfg
    )
    assert r2["pending_action"] == "answer_direct"
    # Critical invariant: chunks NOT leaked from turn 1
    assert (r2.get("retrieved_chunks") or []) == []


# ---------- compaction trigger ----------


async def test_compaction_fires_when_count_threshold_exceeded(
    compiled_agent,
    fake_provider,
    classifier_response,
    monkeypatch,
    complete_summary,
):
    """Pre-seed a thread with many messages via checkpointer, then next
    invocation must run summarize before decide_action.
    """
    from app.services.llm.base import LLMResponse
    from app.config import settings

    monkeypatch.setattr(settings.agent, "compaction_count_threshold", 3)
    monkeypatch.setattr(settings.agent, "k_recent", 2)

    _stub_all_nodes(monkeypatch, fake_provider, classifier_response, action="answer_direct")

    # summarize uses fake_provider.generate() — override to return a real summary
    original_generate = fake_provider.generate

    async def generate_spy(**kwargs):
        # Track that summarize was hit by checking for summarize system prompt
        sys = kwargs.get("system", "")
        if "progressive structured summary" in sys:
            generate_spy.summarize_called = True  # type: ignore[attr-defined]
            return LLMResponse(
                text=complete_summary,
                tokens_input=200,
                tokens_output=50,
                model="fake-sum",
                latency_ms=200,
            )
        return await original_generate(**kwargs)

    generate_spy.summarize_called = False  # type: ignore[attr-defined]
    fake_provider.generate = generate_spy  # type: ignore[assignment]

    cfg = {"configurable": {"thread_id": "compact-1"}, "recursion_limit": 10}

    # Burn 4 turns to exceed count threshold (3)
    for i in range(4):
        await compiled_agent.ainvoke(
            {"messages": [HumanMessage(content=f"msg {i}")], "collection_id": None},
            config=cfg,
        )

    assert generate_spy.summarize_called is True  # type: ignore[attr-defined]
