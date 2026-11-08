"""The WhatsApp webhook answers in the background (audit A11).

The webhook takes a message and returns; the turn runs in a task. A repeated
`MessageSid` and a number over its limit are dropped, one number's messages
are answered one at a time and in order, and no log line carries a number.
Runs the real app over ASGI, without its lifespan; Twilio's signature check
and the turn are replaced.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
import pytest
from twilio.base.exceptions import TwilioRestException

from app.agent.thread_id import whatsapp_thread_id
from app.api.v1.webhooks import whatsapp as webhook
from app.config import settings
from app.main import app
from app.services import twilio_client
from app.services.whatsapp import inbound
from app.services.whatsapp.inbound import SeenMessages

NUMBER = "595981234567"
SENDER = f"whatsapp:+{NUMBER}"


async def _post(path: str, data: dict[str, str]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test/api/v1") as c:
        return await c.post(path, data=data)


def _message(sid: str, body: str = "¿qué dice el contrato?", sender: str = SENDER) -> dict:
    return {"From": sender, "Body": body, "MessageSid": sid}


@pytest.fixture
def signed(monkeypatch):
    """Every request passes Twilio's signature check, and starts with no
    message seen and no turn running."""
    monkeypatch.setattr(webhook, "_verify_signature", lambda *_a: True)
    monkeypatch.setattr(app.state, "agent", object(), raising=False)
    inbound.seen_messages.reset()
    yield
    inbound.seen_messages.reset()


@pytest.fixture
def replies(monkeypatch):
    """The replies the webhook started, in order, instead of running them."""
    started: list[tuple[str, str]] = []

    def spawn(_agent, *, sender, body):
        started.append((sender, body))

    monkeypatch.setattr(webhook, "spawn_reply", spawn)
    return started


async def test_the_webhook_returns_before_the_turn_ends(signed, monkeypatch):
    release = asyncio.Event()
    answered: list[str] = []

    async def slow_reply(_agent, *, sender, body, sessions):
        await release.wait()
        answered.append(body)

    monkeypatch.setattr(inbound, "reply", slow_reply)

    # A webhook that waited for the turn would time out here.
    response = await asyncio.wait_for(_post("/webhook/whatsapp", _message("SM1", "hola")), 5)

    assert response.status_code == 204
    assert answered == []
    (task,) = inbound._running
    release.set()
    await task
    assert answered == ["hola"]
    assert not inbound._running


async def test_a_repeated_message_is_answered_once(signed, replies):
    first = await _post("/webhook/whatsapp", _message("SM1"))
    again = await _post("/webhook/whatsapp", _message("SM1"))
    other = await _post("/webhook/whatsapp", _message("SM2"))

    assert (first.status_code, again.status_code, other.status_code) == (204, 204, 204)
    assert len(replies) == 2


async def test_an_empty_message_starts_no_turn(signed, replies):
    assert (await _post("/webhook/whatsapp", _message("SM1", "   "))).status_code == 204
    assert replies == []


def test_a_seen_message_is_forgotten_after_its_ttl():
    now = [0.0]
    seen = SeenMessages(ttl_s=600, clock=lambda: now[0])

    assert seen.first_time("SM1")
    now[0] = 599
    assert not seen.first_time("SM1")
    now[0] = 600
    assert seen.first_time("SM1")


async def test_a_number_over_its_limit_is_not_answered(signed, replies, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    other = "whatsapp:+595991111111"

    for i in range(3):
        await _post("/webhook/whatsapp", _message(f"SM{i}"))
    await _post("/webhook/whatsapp", _message("SM9", sender=other))

    assert [sender for sender, _ in replies] == [SENDER, SENDER, other]


async def test_one_numbers_messages_are_answered_one_at_a_time_in_order(monkeypatch):
    events: list[str] = []

    async def answer(_agent, _db, *, sender, body, thread_id):
        events.append(f"start {body}")
        await asyncio.sleep(0.01)
        events.append(f"end {body}")

    @asynccontextmanager
    async def sessions():
        yield None

    monkeypatch.setattr(inbound, "_answer", answer)
    other = "whatsapp:+595991111111"

    tasks = [
        inbound.spawn_reply(None, sender=SENDER, body="/coleccion contratos", sessions=sessions),
        inbound.spawn_reply(None, sender=SENDER, body="a", sessions=sessions),
        inbound.spawn_reply(None, sender=other, body="b", sessions=sessions),
        inbound.spawn_reply(None, sender=SENDER, body="c", sessions=sessions),
    ]
    await asyncio.gather(*tasks)

    mine = [e for e in events if not e.endswith(" b")]
    assert mine == [
        "start /coleccion contratos",
        "end /coleccion contratos",
        "start a",
        "end a",
        "start c",
        "end c",
    ]
    # Another number does not wait for this one.
    assert events.index("start b") < events.index("end /coleccion contratos")
    assert inbound._thread_locks == {}


async def test_shutdown_cancels_the_turns_still_running(monkeypatch):
    async def stuck(_agent, _db, **_kw):
        await asyncio.Event().wait()

    @asynccontextmanager
    async def sessions():
        yield None

    monkeypatch.setattr(inbound, "_answer", stuck)
    task = inbound.spawn_reply(None, sender=SENDER, body="hola", sessions=sessions)
    await asyncio.sleep(0)

    assert await inbound.stop_replies(grace_s=0.01) == 1
    assert task.cancelled()
    assert await inbound.stop_replies(grace_s=0.01) == 0


async def test_a_failed_send_logs_twilios_code_not_the_number(monkeypatch, caplog):
    async def rejected(to, body):
        raise TwilioRestException(
            400,
            "/Messages.json",
            f"The 'To' number {SENDER} is not a valid phone number.",
            code=21211,
        )

    monkeypatch.setattr(inbound, "send_whatsapp_message", rejected)

    with caplog.at_level(logging.INFO):
        await inbound._send(SENDER, "hola", whatsapp_thread_id(SENDER))

    assert "twilio_send_failed" in caplog.text
    assert "status=400 code=21211" in caplog.text
    assert NUMBER not in caplog.text


async def test_a_sent_message_logs_no_number_and_reuses_the_client(monkeypatch, caplog):
    made: list[object] = []

    class _Messages:
        def create(self, **kwargs):
            assert kwargs["to"] == SENDER

    class _Client:
        def __init__(self, sid, token):
            made.append(self)
            self.messages = _Messages()

    monkeypatch.setattr("twilio.rest.Client", _Client)
    twilio_client._client.cache_clear()
    try:
        with caplog.at_level(logging.INFO):
            await twilio_client.send_whatsapp_message(SENDER, "hola")
            await twilio_client.send_whatsapp_message(f"+{NUMBER}", "chau")
    finally:
        twilio_client._client.cache_clear()

    assert len(made) == 1
    assert caplog.text.count("twilio_message_sent") == 2
    assert NUMBER not in caplog.text


async def test_a_delivery_status_logs_no_number(signed, caplog):
    with caplog.at_level(logging.INFO):
        response = await _post(
            "/webhook/whatsapp/status",
            {
                "MessageSid": "SM1",
                "MessageStatus": "undelivered",
                "ErrorCode": "63016",
                "To": SENDER,
            },
        )

    assert response.status_code == 204
    assert "status=undelivered error_code=63016" in caplog.text
    assert NUMBER not in caplog.text
