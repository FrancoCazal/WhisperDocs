"""When the query rewrite calls the model (audit A17, N5)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.utils.messages import failed_answer
from app.agent.utils.query_rewrite import rewrite_query_for_retrieval
from app.services.llm.base import LLMResponse


def _rewrites_to(fake_provider, text: str) -> None:
    fake_provider.generate_response = LLMResponse(
        text=text, tokens_input=10, tokens_output=5, model="fake", latency_ms=10
    )


async def test_a_first_message_goes_as_written(fake_provider, mock_llm_service_for):
    mock_llm_service_for("app.agent.utils.query_rewrite", fake_provider)
    # Pronouns and Spanish articles used to send turn 1 to the model.
    current = "¿cómo la configuro y qué parámetros tiene esa función en los workers de ingesta?"

    rewrite = await rewrite_query_for_retrieval([HumanMessage(content=current)], current)

    assert rewrite.query == current
    assert rewrite.error is None
    assert fake_provider.calls == []


async def test_a_follow_up_goes_to_the_model(fake_provider, mock_llm_service_for):
    mock_llm_service_for("app.agent.utils.query_rewrite", fake_provider)
    _rewrites_to(fake_provider, "pgvector Postgres 14 compatibility")
    history = [
        HumanMessage(content="¿qué es pgvector?"),
        AIMessage(content="Una extensión de Postgres [1]."),
        HumanMessage(content="¿se puede usar con Postgres 14?"),
    ]

    rewrite = await rewrite_query_for_retrieval(history, history[-1].text)

    assert rewrite.query == "pgvector Postgres 14 compatibility"
    assert rewrite.error is None
    sent = fake_provider.calls[0]["kwargs"]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]


async def test_a_summary_alone_is_enough_context(fake_provider, mock_llm_service_for):
    mock_llm_service_for("app.agent.utils.query_rewrite", fake_provider)
    current = "and the limits?"

    await rewrite_query_for_retrieval([HumanMessage(content=current)], current, summary="s")

    assert len(fake_provider.calls) == 1


async def test_failed_turns_are_not_context(fake_provider, mock_llm_service_for):
    mock_llm_service_for("app.agent.utils.query_rewrite", fake_provider)
    current = "How is the chunker configured?"
    history = [
        HumanMessage(content=current),
        failed_answer("I had a problem…"),
        HumanMessage(content=current),
    ]

    assert (await rewrite_query_for_retrieval(history, current)).query == current
    assert fake_provider.calls == []


async def test_a_failed_rewrite_searches_as_written_and_says_why(
    fake_provider, mock_llm_service_for
):
    async def refuse(**_):
        raise TimeoutError("read timed out")

    fake_provider.generate = refuse  # type: ignore[assignment]
    mock_llm_service_for("app.agent.utils.query_rewrite", fake_provider, model="rewriter")
    history = [
        HumanMessage(content="¿qué es pgvector?"),
        AIMessage(content="Una extensión de Postgres [1]."),
        HumanMessage(content="¿y sus límites?"),
    ]

    rewrite = await rewrite_query_for_retrieval(history, history[-1].text)

    assert rewrite.query == "¿y sus límites?"
    assert rewrite.error == {"step": "query_rewrite", "kind": "timeout", "model": "rewriter"}
