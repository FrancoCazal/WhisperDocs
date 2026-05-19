"""Async-safe wrapper over the synchronous Twilio REST client.

The official twilio.rest.Client is blocking. Wrap calls in asyncio.to_thread
so FastAPI's event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def send_whatsapp_message(to: str, body: str) -> None:
    """Send a WhatsApp message via Twilio.

    `to` may be 'whatsapp:+...' or a plain phone number — prefix is added if missing.
    """
    from app.config import settings
    from twilio.rest import Client

    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    normalized_to = to if to.startswith("whatsapp:") else f"whatsapp:{to}"

    await asyncio.to_thread(
        client.messages.create,
        from_=settings.twilio_whatsapp_from,
        to=normalized_to,
        body=body,
    )
    logger.info("twilio_message_sent to=%s chars=%d", normalized_to, len(body))
