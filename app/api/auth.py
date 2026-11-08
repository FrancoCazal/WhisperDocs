"""The API key (audit R6).

With `API_KEY` set, every route needs it in the `X-API-Key` header, except
`/health` without `deep` (a liveness probe) and the WhatsApp webhook, which
Twilio calls and which checks Twilio's signature instead. Unset or empty, the
API is open, as a local demo is; the API logs a warning at startup.
"""

from __future__ import annotations

from secrets import compare_digest

from fastapi import Security, status
from fastapi.security import APIKeyHeader

from app.api.errors import APIError
from app.config import settings

API_KEY_HEADER = "X-API-Key"

api_key_header = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description="Needed when the API runs with API_KEY set.",
)


def configured_key() -> str | None:
    key = settings.api_key.get_secret_value() if settings.api_key is not None else ""
    return key or None


def check_api_key(key: str | None) -> None:
    expected = configured_key()
    if expected is None:
        return
    challenge = {"WWW-Authenticate": "ApiKey"}
    if not key:
        raise APIError(
            status.HTTP_401_UNAUTHORIZED,
            "api_key_required",
            f"This API needs an API key in the {API_KEY_HEADER} header.",
            headers=challenge,
        )
    if not compare_digest(key.encode(), expected.encode()):
        raise APIError(
            status.HTTP_401_UNAUTHORIZED,
            "api_key_invalid",
            "The API key is not valid.",
            headers=challenge,
        )


async def require_api_key(key: str | None = Security(api_key_header)) -> None:
    check_api_key(key)
