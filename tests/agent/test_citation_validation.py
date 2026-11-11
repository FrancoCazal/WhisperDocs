"""The semantic citation check (app/agent/utils/citation_validation.py, audit
A19): which citations it flags, what it embeds, and what the turn says when
the check fails."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

import app.agent.nodes.search_docs as search_docs_module
from app.agent.nodes.search_docs import search_docs_node
from app.agent.utils.citation_validation import validate_citations_semantic
from app.services import embedder as embedder_mod
from app.services.llm.base import LLMResponse

NOTHING = {"low_confidence_refs": [], "scores": {}}


class _Embedder:
    """A vector per text; records each batch it is asked for."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.batches: list[list[str]] = []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [self.vectors[t] for t in texts]


def _chunks(*texts: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(text=t) for t in texts]


async def _check(answer: str, chunks, embedder, min_similarity: float = 0.5) -> dict:
    return await validate_citations_semantic(
        answer=answer, included_chunks=chunks, embedder=embedder, min_similarity=min_similarity
    )


async def test_it_flags_a_citation_its_passage_does_not_support():
    embedder = _Embedder(
        {
            "Pgvector stores vectors [1].": [1.0, 0.0],
            "Tino likes pears [2].": [1.0, 0.0],
            # Not unit length: the cosine normalises.
            "pgvector passage": [3.0, 0.0],
            "tino passage": [0.0, 2.0],
        }
    )

    result = await _check(
        "Pgvector stores vectors [1]. Tino likes pears [2].",
        _chunks("pgvector passage", "tino passage"),
        embedder,
    )

    assert result["low_confidence_refs"] == [2]
    assert result["scores"] == {1: pytest.approx(1.0), 2: pytest.approx(0.0)}


@pytest.mark.parametrize(("min_similarity", "flagged"), [(0.6, [1]), (0.4, [])])
async def test_a_citation_used_twice_is_scored_by_its_mean(min_similarity, flagged):
    embedder = _Embedder({"A [1].": [1.0, 0.0], "B [1].": [0.0, 1.0], "passage": [1.0, 0.0]})

    result = await _check("A [1]. B [1].", _chunks("passage"), embedder, min_similarity)

    assert result["scores"] == {1: pytest.approx(0.5)}
    assert result["low_confidence_refs"] == flagged


async def test_it_embeds_the_cited_sentences_and_each_cited_passage_once():
    embedder = _Embedder(
        {"A [2].": [1.0, 0.0], "B [2][1].": [1.0, 0.0], "one": [1.0, 0.0], "two": [1.0, 0.0]}
    )

    result = await _check(
        "No citation here. A [2]. B [2][1]. C [7].", _chunks("one", "two"), embedder
    )

    # One call: the sentences with a valid citation, then the passages by id.
    # [7] is out of range, so "C [7]." is not embedded.
    assert embedder.batches == [["A [2].", "B [2][1].", "one", "two"]]
    assert set(result["scores"]) == {1, 2}


@pytest.mark.parametrize(
    ("answer", "chunks"),
    [
        ("", _chunks("one")),
        ("No citation here.", _chunks("one")),
        ("Out of range [2].", _chunks("one")),
        ("A [1].", []),
    ],
)
async def test_without_a_valid_citation_it_embeds_nothing(answer, chunks):
    embedder = _Embedder({})

    assert await _check(answer, chunks, embedder) == NOTHING
    assert embedder.batches == []


async def test_a_failed_embedding_raises_for_the_caller_to_record():
    class _Down:
        async def embed_batch(self, texts):  # noqa: ARG002
            raise TimeoutError("embeddings timed out")

    with pytest.raises(TimeoutError):
        await _check("A [1].", _chunks("one"), _Down())


# ---------- in the search_docs node ----------


@pytest.fixture
def checked_turn(
    fake_provider,
    mock_llm_service_for,
    mock_embedder,
    mock_retriever,
    mock_session_local,
    rag_chunk,
    monkeypatch,
):
    """search_docs with the check on, two chunks and an answer citing both."""
    mock_retriever([rag_chunk(0), rag_chunk(1)])
    mock_llm_service_for("app.agent.nodes.search_docs", fake_provider)
    fake_provider.generate_response = LLMResponse(
        text="First claim [1]. Second claim [2].",
        tokens_input=10,
        tokens_output=5,
        model="fake-model",
        latency_ms=1,
    )
    agent = search_docs_module.settings.agent
    monkeypatch.setattr(agent, "rag_structured_output", False)
    monkeypatch.setattr(agent, "citation_semantic_validation", True)
    monkeypatch.setattr(agent, "citation_semantic_min_similarity", 0.5)
    monkeypatch.setattr(search_docs_module.settings, "embedding_model", "embedder")

    def _embeds_as(embed_batch):
        monkeypatch.setattr(embedder_mod.EmbeddingService, "embed_batch", embed_batch)

    async def _run():
        return await search_docs_node(
            {"messages": [HumanMessage(content="¿quién es tino?")], "collection_id": None}
        )

    return _embeds_as, _run


async def test_the_turn_keeps_the_flagged_citations(checked_turn):
    embeds_as, run = checked_turn
    vectors = {
        "First claim [1].": [1.0, 0.0],
        "Second claim [2].": [1.0, 0.0],
        "chunk text 0": [1.0, 0.0],
        "chunk text 1": [0.0, 1.0],
    }

    async def embed_batch(self, texts):  # noqa: ARG001
        return [vectors[t] for t in texts]

    embeds_as(embed_batch)

    result = await run()

    assert result["metadata"]["citation_low_confidence_refs"] == [2]
    assert result["metadata"]["citation_min_similarity"] == 0.5
    assert result["messages"][0].additional_kwargs["errors"] == []


async def test_a_failed_check_still_answers_and_says_the_citations_went_unchecked(
    checked_turn,
):
    embeds_as, run = checked_turn

    async def embed_batch(self, texts):  # noqa: ARG001
        raise TimeoutError("embeddings timed out")

    embeds_as(embed_batch)

    result = await run()

    answer = result["messages"][0]
    assert answer.content == "First claim [1]. Second claim [2]."
    assert result["metadata"]["turn_failed"] is False
    assert len(answer.additional_kwargs["citations"]) == 2
    assert "citation_low_confidence_refs" not in result["metadata"]
    error = {"step": "citation_check", "kind": "timeout", "model": "embedder"}
    assert answer.additional_kwargs["errors"] == [error]
    assert result["metadata"]["turn_errors"] == [error]
