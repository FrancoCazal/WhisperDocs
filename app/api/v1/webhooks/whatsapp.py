"""Twilio WhatsApp webhook endpoints.

POST /api/v1/webhook/whatsapp        — inbound messages
POST /api/v1/webhook/whatsapp/status — delivery status callbacks

Signature verification is enforced on both. See phase-4-design §2.11.
"""

from __future__ import annotations

import difflib
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.messages import HumanMessage
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.thread_id import normalize_phone, whatsapp_thread_id
from app.config import settings
from app.db import get_db
from app.models import Collection, Conversation
from app.services.twilio_client import send_whatsapp_message
from app.services.whatsapp.formatter import format_for_whatsapp

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])

_NEW_THREAD_DISCLAIMER = (
    "Iniciamos una conversación nueva. Si querés guardarla, respondé /guardar cuando quieras.\n\n"
)

_HELP_TEXT = (
    "Comandos disponibles:\n"
    "/guardar — Guarda esta conversación (no se borrará automáticamente).\n"
    "/coleccion <nombre> — Cambia la colección activa.\n"
    "/coleccion todas — Consulta todas las colecciones.\n"
    "/ayuda — Muestra este mensaje."
)


# ---------- signature verification ----------


def _verify_signature(request: Request, params: dict, webhook_url: str) -> bool:
    if not settings.twilio_auth_token or not webhook_url:
        return False
    from twilio.request_validator import RequestValidator

    signature = request.headers.get("X-Twilio-Signature", "")
    return RequestValidator(settings.twilio_auth_token).validate(webhook_url, params, signature)


# ---------- slash commands ----------


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


# ---------- grace period check ----------


def _is_expired(conv: Conversation) -> bool:
    if conv.is_saved or settings.conversation_grace_period_days <= 0:
        return False
    last = conv.last_active_at
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=settings.conversation_grace_period_days)
    return last < cutoff


# ---------- endpoints ----------


@router.post("/whatsapp", status_code=204)
async def whatsapp_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> None:
    form = await request.form()
    params = dict(form)

    if not _verify_signature(request, params, settings.twilio_webhook_url):
        logger.warning(
            "twilio_signature_invalid ip=%s",
            getattr(request.client, "host", "unknown"),
        )
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    from_raw = params.get("From", "")
    body = params.get("Body", "").strip()

    if not body:
        return

    from_number = normalize_phone(from_raw)
    thread_id = whatsapp_thread_id(from_raw)

    conv = await db.scalar(select(Conversation).where(Conversation.thread_id == thread_id))

    is_new_thread = False
    if conv is None:
        conv = Conversation(
            thread_id=thread_id,
            user_number=from_number,
            channel="whatsapp",
            collection_id=settings.default_collection_id,
        )
        db.add(conv)
        await db.flush()
        is_new_thread = True
    elif _is_expired(conv):
        # Conversation fell outside grace period and wasn't swept yet — reset memory.
        agent = request.app.state.agent
        try:
            await agent.checkpointer.adelete_thread(thread_id)
        except Exception:
            logger.warning("checkpointer_delete_failed thread_id=%s", thread_id)
        is_new_thread = True

    # Slash commands are handled before the agent — no LLM cost.
    slash_reply = await _handle_slash_command(body, conv, db)
    if slash_reply is not None:
        try:
            await send_whatsapp_message(from_raw, slash_reply)
        except Exception:
            logger.exception("twilio_send_failed thread_id=%s", thread_id)
        return

    agent = request.app.state.agent
    try:
        result = await agent.ainvoke(
            {
                "messages": [HumanMessage(content=body)],
                "collection_id": conv.collection_id,
            },
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
        )
    except Exception:
        logger.exception("agent_invoke_failed thread_id=%s", thread_id)
        try:
            await send_whatsapp_message(
                from_raw, "Disculpá, tuve un problema. Intentá de nuevo en un momento."
            )
        except Exception:
            logger.exception("twilio_error_send_failed thread_id=%s", thread_id)
        return

    conv.last_active_at = func.now()
    await db.commit()

    final = result["messages"][-1]
    answer = final.content if isinstance(final.content, str) else str(final.content)
    citations = result.get("retrieved_chunks") or []

    formatted = format_for_whatsapp(answer, citations or None)
    if is_new_thread:
        formatted = _NEW_THREAD_DISCLAIMER + formatted

    try:
        await send_whatsapp_message(from_raw, formatted)
    except Exception:
        logger.exception("twilio_send_failed thread_id=%s", thread_id)


@router.post("/whatsapp/status", status_code=204)
async def whatsapp_status_webhook(request: Request) -> None:
    form = await request.form()
    params = dict(form)

    if not _verify_signature(request, params, settings.twilio_status_webhook_url):
        logger.warning(
            "twilio_status_signature_invalid ip=%s",
            getattr(request.client, "host", "unknown"),
        )
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    logger.info(
        "twilio_delivery_status message_sid=%s status=%s to=%s",
        params.get("MessageSid"),
        params.get("MessageStatus"),
        params.get("To"),
    )
