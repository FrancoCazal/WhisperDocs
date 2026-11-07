"""Answers the agent writes without a model, in the user's language (audit N8)."""

from collections.abc import Iterator, Sequence
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage

from app.services.lang import reply_language

FixedReply = Literal[
    "no_documents",
    "retrieval_failed",
    "not_enough_information",
    "generation_failed",
    "out_of_scope",
    "clarify",
    "greeting",
]

_REPLIES: dict[FixedReply, dict[str, str]] = {
    "no_documents": {
        "en": "There are no indexed documents to answer this.",
        "es": "No hay documentos indexados para responder esto.",
    },
    "retrieval_failed": {
        "en": "I could not search the documents because of a technical problem. "
        "Please try again in a moment.",
        "es": "No pude buscar en los documentos por un problema técnico. "
        "Probá de nuevo en un momento.",
    },
    "not_enough_information": {
        "en": "I don't have enough information to answer this.",
        "es": "No tengo información suficiente para responder esto.",
    },
    "generation_failed": {
        "en": "I had a problem generating the answer. Please try again in a moment.",
        "es": "Tuve un problema generando la respuesta. Probá de nuevo en un momento.",
    },
    "out_of_scope": {
        "en": "That question is outside the scope of the loaded documents. "
        "Try a question about the content of your collection.",
        "es": "Esa pregunta está fuera del alcance de los documentos cargados. "
        "Probá una pregunta sobre el contenido de tu colección.",
    },
    "clarify": {
        "en": "Can you give me more details about what you need?",
        "es": "¿Podés darme más detalles sobre lo que necesitás?",
    },
    "greeting": {
        "en": "Hi! How can I help you with the documentation?",
        "es": "¡Hola! ¿En qué puedo ayudarte con la documentación?",
    },
}


def _user_texts(messages: Sequence[BaseMessage]) -> Iterator[str]:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            yield m.text


def fixed_reply(key: FixedReply, messages: Sequence[BaseMessage]) -> str:
    """`key`'s text in the language of the user's latest message.

    An undecided message, such as "ok?", takes the language of the user's
    earlier messages.
    """
    return _REPLIES[key][reply_language(_user_texts(messages))]
