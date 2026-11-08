"""Twilio WhatsApp webhook endpoints.

POST /api/v1/webhook/whatsapp        — inbound messages
POST /api/v1/webhook/whatsapp/status — delivery status callbacks

Signature verification is enforced on both. See phase-4-design §2.11.

An inbound message is answered in the background
(`app/services/whatsapp/inbound.py`): Twilio waits at most 15 s, and a turn
can take longer. The webhook returns 204 once the message is taken, dropped
as a repeat, or over its number's limit.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.agent.thread_id import whatsapp_thread_id
from app.api.rate_limit import limiter
from app.config import settings
from app.services.whatsapp.inbound import seen_messages, spawn_reply

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])


def _verify_signature(request: Request, params: dict, webhook_url: str) -> bool:
    if not settings.twilio_auth_token or not webhook_url:
        return False
    from twilio.request_validator import RequestValidator

    signature = request.headers.get("X-Twilio-Signature", "")
    return RequestValidator(settings.twilio_auth_token).validate(webhook_url, params, signature)


@router.post("/whatsapp", operation_id="receive_whatsapp_message", status_code=204)
async def whatsapp_webhook(request: Request) -> None:
    form = await request.form()
    params = dict(form)

    if not _verify_signature(request, params, settings.twilio_webhook_url):
        logger.warning(
            "twilio_signature_invalid ip=%s",
            getattr(request.client, "host", "unknown"),
        )
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    sender = str(params.get("From", ""))
    body = str(params.get("Body", "")).strip()
    if not body:
        return

    message_sid = params.get("MessageSid")
    if message_sid and not seen_messages.first_time(str(message_sid)):
        logger.info("twilio_message_repeated message_sid=%s", message_sid)
        return

    # The questions' limit, per number: each message can be a paid turn.
    thread_id = whatsapp_thread_id(sender)
    limit = settings.rate_limit_per_minute
    if limit > 0 and limiter.hit("whatsapp", thread_id, limit) is not None:
        logger.warning("whatsapp_rate_limited thread_id=%s", thread_id)
        return

    spawn_reply(request.app.state.agent, sender=sender, body=body)


@router.post("/whatsapp/status", operation_id="receive_whatsapp_status", status_code=204)
async def whatsapp_status_webhook(request: Request) -> None:
    form = await request.form()
    params = dict(form)

    if not _verify_signature(request, params, settings.twilio_status_webhook_url):
        logger.warning(
            "twilio_status_signature_invalid ip=%s",
            getattr(request.client, "host", "unknown"),
        )
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    # Not `To`: the recipient's number.
    logger.info(
        "twilio_delivery_status message_sid=%s status=%s error_code=%s",
        params.get("MessageSid"),
        params.get("MessageStatus"),
        params.get("ErrorCode"),
    )
