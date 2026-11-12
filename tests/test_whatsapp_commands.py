"""WhatsApp's slash commands (app/services/whatsapp/inbound.py): each answers
in the language of the name it was sent as. The database is a fake; the
commands against Postgres are in test_whatsapp_reply_db.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.whatsapp import inbound


class _Db:
    def __init__(self, *names: str) -> None:
        self.collections = [SimpleNamespace(id=i, name=n) for i, n in enumerate(names, start=1)]
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1

    async def execute(self, _statement):
        rows = self.collections
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows))


def _conv() -> SimpleNamespace:
    return SimpleNamespace(is_saved=False, collection_id=7)


async def _command(body: str, db: _Db | None = None, conv=None):
    conv = conv or _conv()
    return await inbound._handle_slash_command(body, conv, db or _Db()), conv


@pytest.mark.parametrize(
    ("body", "language"), [("/save", "en"), ("/SAVE ", "en"), ("/guardar", "es")]
)
async def test_save_keeps_the_conversation_and_says_so_in_its_language(body, language):
    reply, conv = await _command(body)

    assert reply == inbound.SAVED[language]
    assert conv.is_saved is True


@pytest.mark.parametrize(("body", "language"), [("/help", "en"), ("/ayuda", "es")])
async def test_help_lists_the_commands_of_its_language(body, language):
    reply, _ = await _command(body)

    assert reply == inbound.HELP_TEXT[language]
    assert ("/save" in reply) is (language == "en")
    assert ("/guardar" in reply) is (language == "es")


@pytest.mark.parametrize(
    ("body", "language"),
    [
        ("/collection all", "en"),
        ("/collection todas", "en"),
        ("/coleccion todas", "es"),
        ("/colección all", "es"),
    ],
)
async def test_every_collection_takes_either_word(body, language):
    reply, conv = await _command(body)

    assert reply == inbound.ALL_COLLECTIONS[language]
    assert conv.collection_id is None


async def test_a_collection_is_found_by_a_close_name():
    reply, conv = await _command("/collection eval golden", _Db("Contracts", "Eval Golden"))

    assert reply == "Now searching: Eval Golden. What would you like to know?"
    assert conv.collection_id == 2


@pytest.mark.parametrize(
    ("body", "db", "reply"),
    [
        (
            "/collection invoices",
            _Db("Contracts", "Eval Golden"),
            "No collection called 'invoices'. Available: Contracts, Eval Golden.",
        ),
        ("/collection invoices", _Db(), "No collection called 'invoices'. Available: (none)."),
        (
            "/coleccion facturas",
            _Db(),
            "No encontré la colección 'facturas'. Disponibles: (ninguna).",
        ),
    ],
)
async def test_a_collection_that_is_not_there_names_the_ones_that_are(body, db, reply):
    answer, conv = await _command(body, db)

    assert answer == reply
    assert conv.collection_id == 7
    assert db.commits == 0


@pytest.mark.parametrize("body", ["What does it say?", "/collection", "/unknown"])
async def test_anything_else_goes_to_the_agent(body):
    reply, conv = await _command(body)

    assert reply is None
    assert (conv.is_saved, conv.collection_id) == (False, 7)
