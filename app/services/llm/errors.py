"""What kind of failure a model, embedding or database call raised.

A turn records one `{step, kind, model}` per call that failed or fell back
(see `app/agent/utils/turn_errors.py`), so the chat can say "classifier:
billing or quota" instead of nothing: with a GCP project without billing,
the classifier and the query rewrite fell back on every turn and the answer
still came, so the outage did not show.

The kind is read, in order, from:
  1. LangChain's provider-neutral `ModelError` types (langchain-google-genai
     raises them);
  2. the HTTP status or error code the SDK exception carries: `status_code`
     (openai), an int `code` (google-genai), botocore's `Error.Code` and
     `HTTPStatusCode` (langchain-aws raises botocore's `ClientError` as is);
  3. the exception's type name.
Each exception in the `__cause__` / `__context__` chain is tried, the outer
one first: `EmbeddingError` and LangChain's errors wrap the SDK's.

A refusal (`auth`, `rate_limit`, HTTP 402) whose message talks about billing,
payment or quota is `billing`: Vertex answers 403 PERMISSION_DENIED "requires
billing to be enabled", OpenAI 429 `insufficient_quota`.
"""

from __future__ import annotations

import json
from typing import Literal

from botocore.exceptions import ClientError
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelRefusalError,
    ModelTimeoutError,
    OutputParserException,
)
from pydantic import ValidationError

ErrorKind = Literal[
    "billing",
    "auth",
    "rate_limit",
    "timeout",
    "unavailable",
    "invalid_request",
    "bad_response",
    "unknown",
]

_MODEL_ERRORS: tuple[tuple[type[BaseException], ErrorKind], ...] = (
    (ModelAuthenticationError, "auth"),
    (ModelPermissionDeniedError, "auth"),
    (ModelRateLimitError, "rate_limit"),
    (ModelTimeoutError, "timeout"),
    (ModelConnectionError, "unavailable"),
    (ModelAPIError, "unavailable"),
    (ContextOverflowError, "invalid_request"),
    (ModelInvalidRequestError, "invalid_request"),
    (ModelNotFoundError, "invalid_request"),
    (ModelRefusalError, "bad_response"),
    (OutputParserException, "bad_response"),
)

# botocore error codes, lowercased: ConverseStream reports an error that
# arrives as a stream event with the event name as the code, e.g.
# "throttlingException", and no HTTP status.
_AWS_CODES: dict[str, ErrorKind] = {
    "accessdeniedexception": "auth",
    "unrecognizedclientexception": "auth",
    "invalidsignatureexception": "auth",
    "expiredtokenexception": "auth",
    "invalidclienttokenid": "auth",
    "throttlingexception": "rate_limit",
    "servicequotaexceededexception": "rate_limit",
    "toomanyrequestsexception": "rate_limit",
    "modeltimeoutexception": "timeout",
    "serviceunavailableexception": "unavailable",
    "internalserverexception": "unavailable",
    "modelstreamerrorexception": "unavailable",
    "modelnotreadyexception": "unavailable",
    "validationexception": "invalid_request",
    "resourcenotfoundexception": "invalid_request",
    "modelerrorexception": "bad_response",
}

# Not "quota exceeded": Vertex's per-minute 429 says it, and that is a rate
# limit. OpenAI's `insufficient_quota` message also says "billing details".
_BILLING_WORDS = ("billing", "payment", "insufficient_quota")

# The parse failures the providers raise as ValueError
# ("bedrock_structured_output_parse_failed: ...").
_PARSE_FAILED = "structured_output_parse_failed"


def _chain(exc: BaseException, limit: int = 6) -> list[BaseException]:
    out: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and cur not in out and len(out) < limit:
        out.append(cur)
        cur = cur.__cause__ or cur.__context__
    return out


def _status(exc: BaseException) -> int | None:
    if isinstance(exc, ClientError):
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return status if isinstance(status, int) else None
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        # bool is an int; openai's `code` is a string ("insufficient_quota").
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value < 600:
            return value
    return None


def _by_status(status: int) -> ErrorKind | None:
    if status == 402:
        return "billing"
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limit"
    if status in (408, 504):
        return "timeout"
    if status >= 500:
        return "unavailable"
    if status >= 400:
        return "invalid_request"
    return None


def _by_name(exc: BaseException) -> ErrorKind | None:
    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or "Timeout" in name:
        return "timeout"
    if "Credential" in name:
        return "auth"
    if isinstance(exc, ConnectionError) or "Connection" in name:
        return "unavailable"
    if isinstance(exc, ValidationError | json.JSONDecodeError) or _PARSE_FAILED in str(exc):
        return "bad_response"
    return None


def _kind_of(exc: BaseException) -> tuple[ErrorKind | None, int | None]:
    for cls, kind in _MODEL_ERRORS:
        if isinstance(exc, cls):
            return kind, _status(exc)
    if isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code", "")).lower()
        if code in _AWS_CODES:
            return _AWS_CODES[code], _status(exc)
    status = _status(exc)
    if status is not None:
        by_status = _by_status(status)
        if by_status is not None:
            return by_status, status
    return _by_name(exc), status


def classify_error(exc: BaseException) -> ErrorKind:
    """The kind of failure `exc` reports; `unknown` when nothing says."""
    chain = _chain(exc)
    kind: ErrorKind | None = None
    status: int | None = None
    for e in chain:
        kind, status = _kind_of(e)
        if kind is not None:
            break
    if kind in ("auth", "rate_limit", "billing") or status == 402:
        text = " ".join(str(e) for e in chain).lower()
        if any(word in text for word in _BILLING_WORDS):
            return "billing"
    return kind or "unknown"


# Why a model or embedding call failed, as a client reads it.
KIND_TEXT: dict[ErrorKind, str] = {
    "billing": "the provider refused it over billing or quota",
    "auth": "the provider refused the credentials",
    "rate_limit": "the provider's rate limit was hit",
    "timeout": "it timed out",
    "unavailable": "the provider could not be reached",
    "invalid_request": "the provider rejected the request",
    "bad_response": "the model's answer could not be read",
    "unknown": "the API's logs have the details",
}


def failure_text(exc: BaseException) -> str:
    """Why a call failed, without the exception's own text, which can carry
    a provider's or the database's internals (audit R9)."""
    return KIND_TEXT[classify_error(exc)]
