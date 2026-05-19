"""Unit tests for each agent node with mocked LLM provider + retriever."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes.answer_direct import answer_direct_node
from app.agent.nodes.ask_clarification import ask_clarification_node
from app.agent.nodes.check_compaction import check_compaction_node
from app.agent.nodes.decide_action import decide_action_node
from app.agent.nodes.search_docs import search_docs_node
from app.agent.nodes.summarize import summarize_node


# ---------- check_compaction ----------


async def test_check_compaction_below_threshold_no_compaction():
    state = {"messages": [HumanMessage(content="hola")]}
    result = await check_compaction_node(state)
    assert result["pending_action"] is None
    assert result["retrieved_chunks"] is None
    assert result["metadata"] == {}


async def test_check_compaction_count_threshold_triggers(monkeypatch):
    # Force count threshold low so 3 messages already trigger.
    from app.config import settings

    monkeypatch.setattr(settings.agent, "compaction_count_threshold", 2)
    state = {
        "messages": [
            HumanMessage(content="a"),
            AIMessage(content="b"),
            HumanMessage(content="c"),
        ]
    }
    result = await check_compaction_node(state)
    assert result["pending_action"] == "summarize"


async def test_check_compaction_token_threshold_triggers(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings.agent, "compaction_token_threshold", 5)
    state = {"messages": [HumanMessage(content="this is a longer message " * 10)]}
    result = await check_compaction_node(state)
    assert result["pending_action"] == "summarize"


async def test_check_compaction_resets_transient_fields():
    # Even if state carries stale transient fields, node must blank them.
    state = {
        "messages": [HumanMessage(content="hi")],
        "retrieved_chunks": [{"chunk_id": 1}],
        "metadata": {"generation_model": "old"},
    }
    result = await check_compaction_node(state)
    assert result["retrieved_chunks"] is None
    assert result["metadata"] == {}


# ---------- decide_action ----------


async def test_decide_action_classifies_search(
    fake_provider, mock_llm_service_for, classifier_response
):
    fake_provider.structured_response = classifier_response("search_docs", "technical", "high")
    mock_llm_service_for("app.agent.nodes.decide_action", fake_provider)

    state = {"messages": [HumanMessage(content="¿qué es pgvector?")]}
    result = await decide_action_node(state)

    assert result["pending_action"] == "search_docs"
    assert result["metadata"]["classifier_reason"] == "technical"
    assert result["metadata"]["classifier_confidence"] == "high"


async def test_decide_action_classifies_direct(
    fake_provider, mock_llm_service_for, classifier_response
):
    fake_provider.structured_response = classifier_response("answer_direct", "greeting", "high")
    mock_llm_service_for("app.agent.nodes.decide_action", fake_provider)

    result = await decide_action_node({"messages": [HumanMessage(content="hola")]})
    assert result["pending_action"] == "answer_direct"


async def test_decide_action_falls_back_on_provider_error(
    fake_provider, mock_llm_service_for, monkeypatch
):
    async def boom(**_):
        raise RuntimeError("provider exploded")

    fake_provider.generate_structured = boom  # type: ignore[assignment]
    mock_llm_service_for("app.agent.nodes.decide_action", fake_provider)

    result = await decide_action_node({"messages": [HumanMessage(content="hi")]})
    # Fallback on classifier error is now `ask_clarification` (more conservative
    # than the previous `answer_direct`, which could hallucinate).
    assert result["pending_action"] == "ask_clarification"
    assert result["metadata"]["classifier_confidence"] == "low"


async def test_decide_action_low_confidence_downgrades_to_clarification(
    fake_provider, mock_llm_service_for, classifier_response
):
    fake_provider.structured_response = classifier_response("search_docs", "uncertain", "low")
    mock_llm_service_for("app.agent.nodes.decide_action", fake_provider)

    result = await decide_action_node({"messages": [HumanMessage(content="¿algo?")]})
    # Predicted action was search_docs with low confidence → downgraded.
    assert result["pending_action"] == "ask_clarification"
    assert result["metadata"]["classifier_confidence"] == "low"


async def test_decide_action_classifies_out_of_scope(
    fake_provider, mock_llm_service_for, classifier_response
):
    fake_provider.structured_response = classifier_response("out_of_scope", "off-topic", "high")
    mock_llm_service_for("app.agent.nodes.decide_action", fake_provider)

    result = await decide_action_node(
        {"messages": [HumanMessage(content="¿qué clima hace en Madrid?")]}
    )
    assert result["pending_action"] == "out_of_scope"


# ---------- out_of_scope ----------


async def test_out_of_scope_node_returns_canned_message():
    from app.agent.nodes.out_of_scope import out_of_scope_node

    result = await out_of_scope_node({"messages": [HumanMessage(content="clima madrid?")]})
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert "fuera del alcance" in msg.content.lower()
    # No LLM call → no model recorded.
    assert result["metadata"]["generation_model"] is None


# ---------- ask_clarification ----------


async def test_ask_clarification_returns_ai_message(fake_provider, mock_llm_service_for):
    from app.services.llm.base import LLMResponse

    fake_provider.generate_response = LLMResponse(
        text="¿Qué querés saber exactamente?",
        tokens_input=10,
        tokens_output=6,
        model="fake",
        latency_ms=50,
    )
    mock_llm_service_for("app.agent.nodes.ask_clarification", fake_provider)

    result = await ask_clarification_node({"messages": [HumanMessage(content="ayuda")]})
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "querés saber" in result["messages"][0].content


async def test_ask_clarification_falls_back_on_error(fake_provider, mock_llm_service_for):
    async def boom(**_):
        raise RuntimeError("down")

    fake_provider.generate = boom  # type: ignore[assignment]
    mock_llm_service_for("app.agent.nodes.ask_clarification", fake_provider)

    result = await ask_clarification_node({"messages": [HumanMessage(content="?")]})
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content  # non-empty fallback


# ---------- answer_direct ----------


async def test_answer_direct_returns_ai_message(fake_provider, mock_llm_service_for):
    from app.services.llm.base import LLMResponse

    fake_provider.generate_response = LLMResponse(
        text="¡Hola! ¿En qué te ayudo?",
        tokens_input=8,
        tokens_output=6,
        model="fake",
        latency_ms=80,
    )
    mock_llm_service_for("app.agent.nodes.answer_direct", fake_provider)

    result = await answer_direct_node({"messages": [HumanMessage(content="hola")]})
    assert isinstance(result["messages"][0], AIMessage)
    assert "Hola" in result["messages"][0].content


# ---------- search_docs ----------


async def test_search_docs_with_chunks(
    fake_provider,
    mock_llm_service_for,
    mock_embedder,
    mock_retriever,
    mock_session_local,
    rag_chunk,
    monkeypatch,
):
    from app.services.llm.base import LLMResponse

    mock_retriever([rag_chunk(0), rag_chunk(1, rel=0.7)])

    fake_provider.generate_response = LLMResponse(
        text="La respuesta es [1].",
        tokens_input=100,
        tokens_output=20,
        model="fake-rag",
        latency_ms=300,
    )
    mock_llm_service_for("app.agent.nodes.search_docs", fake_provider)
    monkeypatch.setattr(
        "app.agent.utils.query_rewrite.llm_service_for",
        lambda _n: (fake_provider, "fake", 0.0),
    )

    state = {
        "messages": [HumanMessage(content="¿quién es tino?")],
        "collection_id": 3,
    }
    result = await search_docs_node(state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert len(result["retrieved_chunks"]) == 2
    assert result["retrieved_chunks"][0]["document_title"] == "tino_lore.md"
    assert result["metadata"]["retrieval_chunks_count"] == 2
    assert result["metadata"]["retrieval_top_relevance"] == 0.9
    assert result["metadata"]["collection_id_active"] == 3


async def test_search_docs_empty_retrieval(
    fake_provider,
    mock_llm_service_for,
    mock_embedder,
    mock_retriever,
    mock_session_local,
    monkeypatch,
):
    from app.services.llm.base import LLMResponse

    mock_retriever([])
    fake_provider.generate_response = LLMResponse(
        text="",
        tokens_input=50,
        tokens_output=5,
        model="fake-rag",
        latency_ms=100,
    )
    mock_llm_service_for("app.agent.nodes.search_docs", fake_provider)
    monkeypatch.setattr(
        "app.agent.utils.query_rewrite.llm_service_for",
        lambda _n: (fake_provider, "fake", 0.0),
    )

    state = {
        "messages": [HumanMessage(content="¿algo?")],
        "collection_id": None,
    }
    result = await search_docs_node(state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "No tengo información suficiente" in result["messages"][0].content
    assert result["retrieved_chunks"] == []
    assert result["metadata"]["retrieval_chunks_count"] == 0
    assert result["metadata"]["retrieval_top_relevance"] == 0.0


# ---------- summarize ----------


async def test_summarize_early_return_when_short():
    # len(messages) <= K (default 8) → no-op
    state = {"messages": [HumanMessage(content="hi")] * 3}
    result = await summarize_node(state)
    assert result == {"pending_action": None}


async def test_summarize_writes_progressive_summary(
    fake_provider, mock_llm_service_for, monkeypatch
):
    from app.services.llm.base import LLMResponse
    from app.config import settings

    monkeypatch.setattr(settings.agent, "k_recent", 2)

    # 6 messages, K=2 → absorb first 4 into summary, do not delete any.
    messages = []
    for i in range(6):
        m = HumanMessage(content=f"msg {i}") if i % 2 == 0 else AIMessage(content=f"ai {i}")
        m.id = f"msg-{i}"
        messages.append(m)

    fake_provider.generate_response = LLMResponse(
        text="## Objetivo\nUser asked about X\n## Decisiones\n(ninguna)",
        tokens_input=400,
        tokens_output=80,
        model="fake-sum",
        latency_ms=400,
    )
    mock_llm_service_for("app.agent.nodes.summarize", fake_provider)

    result = await summarize_node({"messages": messages, "summary": None})

    assert result["summary"].startswith("## Objetivo")
    assert result["pending_action"] is None
    assert result["summary_covered_count"] == 4
    # Literal transcript preserved — node returns no "messages" patch.
    assert "messages" not in result


async def test_summarize_skips_when_already_covered(
    fake_provider, mock_llm_service_for, monkeypatch
):
    from app.services.llm.base import LLMResponse
    from app.config import settings

    monkeypatch.setattr(settings.agent, "k_recent", 2)

    # 6 messages with summary_covered_count already at 4 → trailing 2 must
    # stay outside summary, nothing left to compact → no-op.
    messages = []
    for i in range(6):
        m = HumanMessage(content=f"msg {i}") if i % 2 == 0 else AIMessage(content=f"ai {i}")
        m.id = f"msg-{i}"
        messages.append(m)

    fake_provider.generate_response = LLMResponse(
        text="should not be called", tokens_input=0, tokens_output=0, model="x", latency_ms=0
    )
    mock_llm_service_for("app.agent.nodes.summarize", fake_provider)

    result = await summarize_node(
        {"messages": messages, "summary": "prior", "summary_covered_count": 4}
    )
    assert result == {"pending_action": None}


async def test_summarize_graceful_on_llm_error(fake_provider, mock_llm_service_for, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings.agent, "k_recent", 2)

    async def boom(**_):
        raise RuntimeError("provider failed")

    fake_provider.generate = boom  # type: ignore[assignment]
    mock_llm_service_for("app.agent.nodes.summarize", fake_provider)

    messages = []
    for i in range(6):
        m = HumanMessage(content=f"msg {i}") if i % 2 == 0 else AIMessage(content=f"ai {i}")
        m.id = f"msg-{i}"
        messages.append(m)

    # Must NOT raise — failure returns empty patch, history intact.
    result = await summarize_node({"messages": messages, "summary": None})
    assert result == {"pending_action": None}


async def test_ask_clarification_preserves_prior_metadata(fake_provider, mock_llm_service_for):
    mock_llm_service_for("app.agent.nodes.ask_clarification", fake_provider)
    state = {
        "messages": [HumanMessage(content="?")],
        "metadata": {"classifier_reason": "vague"},
    }
    result = await ask_clarification_node(state)
    # Prior fields preserved + new fields merged
    assert result["metadata"]["classifier_reason"] == "vague"
    assert "generation_model" in result["metadata"]
