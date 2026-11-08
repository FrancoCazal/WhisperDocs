"""Inbound WhatsApp messages, answered in the background (audit A11).

Twilio waits at most 15 s for the webhook, and a turn's p95 was 9.1 s. So the
webhook only checks the signature, drops a message it already took, counts
it against the number's limit and returns 204; the turn runs here, in a task
this process keeps until it ends.

- **A message can arrive twice.** By default Twilio retries a webhook once
  when the connection fails, and the webhook URL can ask for retries on a
  timeout or a 5xx (`#rp=`) or name a fallback URL. Each message has its
  `MessageSid`, and `SeenMessages` drops a second delivery.
- **One number's messages are answered one at a time,** in the order the
  webhook took them: two turns on a thread at once would each read its
  memory before the other wrote it, and a `/coleccion` sent before a
  question must apply to it.
- **This process's memory,** like the rate limits: the API runs as one
  uvicorn process. A restart forgets the ids it saw, and `stop_replies`
  cancels the turns still running after a grace period.
- **No phone number in a log line.** The thread id is a UUID5 of the number.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import time
from collections import Counter, OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.messages import HumanMessage
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.thread_id import normalize_phone, thread_title, whatsapp_thread_id
from app.agent.utils.messages import stamped
from app.agent.utils.turn_errors import failure_kind
from app.config import settings
from app.db import SessionLocal
from app.metrics import agent_turn_duration_seconds
from app.models import Collection, Conversation
from app.services.twilio_client import failure_fields, send_whatsapp_message
from app.services.whatsapp.formatter import format_for_whatsapp
from app.telemetry import AGENT_WORKFLOW, error_type, workflow

logger = logging.getLogger(__name__)

Sessions = Callable[[], AbstractAsyncContextManager[AsyncSession]]

# Longer than any redelivery: Twilio's retries and a fallback URL come within
# the 15 s it waits.
SEEN_TTL_S = 600.0

NEW_THREAD_DISCLAIMER = (
    "Iniciamos una conversación nueva. Si querés guardarla, respondé /guardar cuando quieras.\n\n"
)
APOLOGY = "Disculpá, tuve un problema. Intentá de nuevo en un momento."

_HELP_TEXT = (
    "Comandos disponibles:\n"
    "/guardar — Guarda esta conversación (no se borrará automáticamente).\n"
    "/coleccion <nombre> — Cambia la colección activa.\n"
    "/coleccion todas — Consulta todas las colecciones.\n"
    "/ayuda — Muestra este mensaje."
)


class SeenMessages:
    """The `MessageSid`s taken in the last `ttl_s` seconds."""

    def __init__(self, ttl_s: float = SEEN_TTL_S, clock: Callable[[], float] = time.monotonic):
        self._ttl_s = ttl_s
        self._clock = clock
        self._seen: OrderedDict[str, float] = OrderedDict()

    def first_time(self, message_sid: str) -> bool:
        """Take `message_sid`; False when it was already taken."""
        now = self._clock()
        while self._seen and now - next(iter(self._seen.values())) >= self._ttl_s:
            self._seen.popitem(last=False)
        if message_sid in self._seen:
            return False
        self._seen[message_sid] = now
        return True

    def reset(self) -> None:
        self._seen.clear()


seen_messages = SeenMessages()

# A lock per thread with a turn running or waiting, dropped with the last one.
# asyncio.Lock wakes its waiters in the order they came.
_thread_locks: dict[str, asyncio.Lock] = {}
_thread_turns: Counter[str] = Counter()
_running: set[asyncio.Task[None]] = set()


@asynccontextmanager
async def _one_at_a_time(thread_id: str) -> AsyncIterator[None]:
    lock = _thread_locks.setdefault(thread_id, asyncio.Lock())
    _thread_turns[thread_id] += 1
    try:
        async with lock:
            yield
    finally:
        _thread_turns[thread_id] -= 1
        if not _thread_turns[thread_id]:
            del _thread_turns[thread_id]
            del _thread_locks[thread_id]


def spawn_reply(
    agent: Any, *, sender: str, body: str, sessions: Sessions = SessionLocal
) -> asyncio.Task[None]:
    """Answer `body` from `sender` in the background. The task is kept until
    it ends."""
    task = asyncio.create_task(
        reply(agent, sender=sender, body=body, sessions=sessions), name="whatsapp-reply"
    )
    _running.add(task)
    task.add_done_callback(_running.discard)
    return task


async def stop_replies(grace_s: float = 5.0) -> int:
    """On shutdown: wait up to `grace_s` for the turns running, then cancel
    the rest. Returns how many were cancelled; their senders get no answer."""
    if not _running:
        return 0
    _done, pending = await asyncio.wait(set(_running), timeout=grace_s)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(pending, timeout=2.0)
    return len(pending)


async def reply(agent: Any, *, sender: str, body: str, sessions: Sessions = SessionLocal) -> None:
    """Answer one message: a slash command, or a turn of the agent."""
    thread_id = whatsapp_thread_id(sender)
    try:
        async with _one_at_a_time(thread_id), sessions() as db:
            await _answer(agent, db, sender=sender, body=body, thread_id=thread_id)
    except Exception:
        # The engine hides statement parameters (app/db.py), so the
        # traceback holds no number.
        logger.exception("whatsapp_reply_failed thread_id=%s", thread_id)


async def _send(to: str, text: str, thread_id: str) -> None:
    try:
        await send_whatsapp_message(to, text)
    except Exception as exc:
        logger.error("twilio_send_failed thread_id=%s %s", thread_id, failure_fields(exc))


async def _answer(agent: Any, db: AsyncSession, *, sender: str, body: str, thread_id: str) -> None:
    conv = await db.scalar(select(Conversation).where(Conversation.thread_id == thread_id))

    is_new_thread = False
    if conv is None:
        conv = Conversation(
            thread_id=thread_id,
            user_number=normalize_phone(sender),
            channel="whatsapp",
            collection_id=settings.default_collection_id,
        )
        db.add(conv)
        await db.flush()
        is_new_thread = True
    elif _is_expired(conv):
        # Conversation fell outside grace period and wasn't swept yet — reset memory.
        try:
            await agent.checkpointer.adelete_thread(thread_id)
        except Exception:
            logger.warning("checkpointer_delete_failed thread_id=%s", thread_id)
        is_new_thread = True

    # Slash commands are handled before the agent — no LLM cost.
    slash_reply = await _handle_slash_command(body, conv, db)
    if slash_reply is not None:
        await _send(sender, slash_reply, thread_id)
        return

    # A fresh memory is a new thread: its first message is the title.
    if is_new_thread or conv.title is None:
        conv.title = thread_title(body)

    turn_started = time.perf_counter()
    with workflow(
        AGENT_WORKFLOW,
        conversation_id=thread_id,
        channel="whatsapp",
        collection_id=conv.collection_id,
    ) as turn:
        try:
            result = await agent.ainvoke(
                {
                    "messages": [HumanMessage(content=body, additional_kwargs=stamped())],
                    "collection_id": conv.collection_id,
                },
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
            )
        except Exception as exc:
            logger.exception("agent_invoke_failed thread_id=%s", thread_id)
            turn.fail(error_type(exc))
            result = None
        else:
            turn.outcome(result.get("pending_action"), failure_kind(result.get("metadata") or {}))
    if result is None:
        await _send(sender, APOLOGY, thread_id)
        return

    agent_turn_duration_seconds.labels(action=result.get("pending_action") or "unknown").observe(
        time.perf_counter() - turn_started
    )
    conv.last_active_at = func.now()
    await db.commit()

    final = result["messages"][-1]
    answer = final.content if isinstance(final.content, str) else str(final.content)
    citations = result.get("retrieved_chunks") or []

    formatted = format_for_whatsapp(answer, citations or None)
    if is_new_thread:
        formatted = NEW_THREAD_DISCLAIMER + formatted
    await _send(sender, formatted, thread_id)


async def _handle_slash_command(body: str, conv: Conversation, db: AsyncSession) -> str | None:
    """Parse slash commands. Returns reply text if handled, None otherwise."""
    lower = body.strip().lower()

    if lower in ("/guardar", "/save"):
        conv.is_saved = True
        await db.commit()
        return "Conversación guardada. No se borrará automáticamente."

    if lower in ("/ayuda", "/help"):
        return _HELP_TEXT

    for prefix in ("/coleccion ", "/colección ", "/collection "):
        if lower.startswith(prefix):
            name = body.strip()[len(prefix) :].strip()
            if name.lower() == "todas":
                conv.collection_id = None
                await db.commit()
                return "Consultando todas las colecciones."

            cols = (await db.execute(select(Collection))).scalars().all()
            name_map = {c.name.lower(): c for c in cols}
            matches = difflib.get_close_matches(name.lower(), name_map.keys(), n=1, cutoff=0.6)
            if matches:
                matched = name_map[matches[0]]
                conv.collection_id = matched.id
                await db.commit()
                return f"Ahora consultando: {matched.name}. ¿Qué querés saber?"
            available = ", ".join(c.name for c in cols) or "(ninguna)"
            return f"No encontré la colección '{name}'. Disponibles: {available}."

    return None


def _is_expired(conv: Conversation) -> bool:
    if conv.is_saved or settings.conversation_grace_period_days <= 0:
        return False
    last = conv.last_active_at
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    cutoff = datetime.now(tz=UTC) - timedelta(days=settings.conversation_grace_period_days)
    return last < cutoff
