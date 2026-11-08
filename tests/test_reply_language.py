"""Language of the answers written without a model (audit N8)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.utils.fixed_replies import fixed_reply
from app.services.lang import reply_language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("¿qué clima hace en Madrid?", "es"),  # marks
        ("como configuro los limites de la API", "es"),  # no accents, Spanish words
        ("How do I configure the rate limits?", "en"),
        ("What does the ingestion worker do when a job fails?", "en"),
    ],
)
def test_reply_language_of_one_message(text, expected):
    assert reply_language([text]) == expected


def test_an_undecided_message_takes_the_language_of_earlier_ones():
    # Newest first: "ok?" settles nothing, the earlier question does.
    assert reply_language(["ok?", "¿cómo se configura el chunker?"]) == "es"
    assert reply_language(["ok?", "How is the chunker configured?"]) == "en"


def test_english_when_nothing_settles_it():
    assert reply_language(["ok?"]) == "en"
    assert reply_language([]) == "en"


def test_fixed_reply_follows_the_latest_user_message():
    thread = [
        HumanMessage(content="¿qué es pgvector?"),
        AIMessage(content="Es una extensión [1]."),
        HumanMessage(content="What about the HNSW index?"),
    ]
    assert fixed_reply("no_documents", thread) == "There are no indexed documents to answer this."
    assert fixed_reply("no_documents", thread[:1]) == (
        "No hay documentos indexados para responder esto."
    )
