"""A turn through the real graph is one trace: the turn, its nodes, and the
calls each node made (app/telemetry.py, audit M-3).

The models are fakes behind the real `TrackedProvider`, and the retrieval
pipeline runs with the database search replaced, so its span is the real one.
"""

from __future__ import annotations

import dataclasses

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_agent_graph
from app.config import settings
from app.services.llm.base import LLMResponse
from app.services.llm.tracked import TrackedProvider
from app.services.reranker import NoopReranker
from app.telemetry import AGENT_WORKFLOW, workflow

SONNET = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
GEMINI = "gemini-3.8-flash"


async def test_a_turn_is_one_trace_of_its_nodes_and_calls(
    spans,
    fake_provider,
    classifier_response,
    mock_embedder,
    mock_session_local,
    rag_chunk,
    monkeypatch,
):
    parsed, classified = classifier_response("search_docs")
    fake_provider.structured_response = (parsed, dataclasses.replace(classified, model=GEMINI))
    fake_provider.generate_response = LLMResponse(
        text="Respuesta [1].", tokens_input=900, tokens_output=60, model=SONNET, latency_ms=1
    )
    services = {
        "app.agent.nodes.decide_action": ("classifier", GEMINI),
        "app.agent.utils.query_rewrite": ("query_rewrite", GEMINI),
        "app.agent.nodes.search_docs": ("rag_generation", SONNET),
    }
    for module, (role, model) in services.items():
        provider = TrackedProvider(fake_provider, role)  # type: ignore[arg-type]
        monkeypatch.setattr(
            f"{module}.llm_service_for", lambda _n, _p=provider, _m=model: (_p, _m, 0.0)
        )

    async def hybrid_retrieve(self, **_kwargs):
        return [rag_chunk(0), rag_chunk(1, rel=0.75)]

    monkeypatch.setattr("app.services.retriever.VectorRetriever.hybrid_retrieve", hybrid_retrieve)
    monkeypatch.setattr("app.services.retrieval_pipeline.get_reranker", NoopReranker)
    monkeypatch.setattr(settings, "mmr_enabled", False)
    monkeypatch.setattr(settings.agent, "streaming_tokens", True)

    agent = build_agent_graph().compile(checkpointer=MemorySaver())
    with workflow(AGENT_WORKFLOW, conversation_id="t-trace", collection_id=3) as turn:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="¿qué es pgvector?")], "collection_id": 3},
            config={"configurable": {"thread_id": "t-trace", "top_k": 2}, "recursion_limit": 10},
        )
        turn.outcome(result["pending_action"], None)

    finished = spans.get_finished_spans()
    (root,) = [s for s in finished if s.parent is None]
    assert {s.context.trace_id for s in finished} == {root.context.trace_id}

    def children(parent):
        return [s for s in finished if s.parent and s.parent.span_id == parent.context.span_id]

    nodes = {s.name: s for s in children(root)}
    assert {"check_compaction", "decide_action", "search_docs"} <= set(nodes)

    (classifier,) = children(nodes["decide_action"])
    assert classifier.name == f"chat {GEMINI}"
    assert classifier.attributes["whisperdocs.role"] == "classifier"
    assert classifier.attributes["gen_ai.conversation.id"] == "t-trace"

    under_search = {s.name: s for s in children(nodes["search_docs"])}
    retrieval = under_search["retrieval"]
    # The node got `config`: the request's top_k reached the pipeline.
    assert retrieval.attributes["gen_ai.retrieval.top_k"] == 2
    assert retrieval.attributes["gen_ai.data_source.id"] == "3"
    assert retrieval.attributes["whisperdocs.retrieval.chunks"] == 2
    generation = under_search[f"chat {SONNET}"]
    assert generation.attributes["gen_ai.request.stream"] is True
    assert generation.attributes["gen_ai.usage.input_tokens"] == 900

    calls = [s for s in finished if s.name.startswith("chat ")]
    assert root.attributes["gen_ai.usage.input_tokens"] == sum(
        c.attributes["gen_ai.usage.input_tokens"] for c in calls
    )
    assert root.attributes["whisperdocs.cost_usd"] == round(
        sum(c.attributes["gen_ai.usage.cost"] for c in calls), 6
    )
    assert root.attributes["whisperdocs.action"] == "search_docs"
