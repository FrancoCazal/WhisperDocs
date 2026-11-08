"""Async-safe wrapper over the synchronous Twilio REST client.

The official twilio.rest.Client is blocking. Wrap calls in asyncio.to_thread
so FastAPI's event loop is never blocked.

No log line here carries a phone number (audit A11): the callers log the
thread id, a UUID5 of the number, and `failure_fields` keeps Twilio's error
message out, since it can quote the number.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twilio.rest import Client

logger = logging.getLogger(__name__)


@functools.cache
def _client(account_sid: str, auth_token: str) -> Client:
    """One client per account, reused: Twilio's keeps its HTTP connections."""
    from twilio.rest import Client

    return Client(account_sid, auth_token)


async def send_whatsapp_message(to: str, body: str) -> None:
    """Send a WhatsApp message via Twilio.

    `to` may be 'whatsapp:+...' or a plain phone number — prefix is added if missing.
    """
    from app.config import settings

    client = _client(settings.twilio_account_sid, settings.twilio_auth_token)
    normalized_to = to if to.startswith("whatsapp:") else f"whatsapp:{to}"

    await asyncio.to_thread(
        client.messages.create,
        from_=settings.twilio_whatsapp_from,
        to=normalized_to,
        body=body,
    )
    logger.info("twilio_message_sent chars=%d", len(body))


def failure_fields(exc: BaseException) -> str:
    """How a send failed, for a log line: the exception's class and, for
    Twilio's own errors, the HTTP status and Twilio's error code. Not the
    message, which can quote the number ("The 'To' number ... is not a valid
    phone number")."""
    status = getattr(exc, "status", None)
    code = getattr(exc, "code", None)
    return f"error={type(exc).__name__} status={status} code={code}"
