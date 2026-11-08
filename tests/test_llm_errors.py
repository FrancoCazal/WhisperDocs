"""classify_error on the exceptions the SDKs raise, built as they build them."""

from __future__ import annotations

import httpx2
import openai
import pytest
from botocore.exceptions import (
    ClientError as BotoClientError,
)
from botocore.exceptions import (
    EndpointConnectionError,
    NoCredentialsError,
    ReadTimeoutError,
)
from google.genai.errors import ClientError as GenaiClientError
from google.genai.errors import ServerError as GenaiServerError
from langchain_core.exceptions import OutputParserException
from langchain_google_genai.chat_models import (
    GooglePermissionDeniedError,
    GoogleRateLimitError,
)
from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import OperationalError

from app.services.embeddings.base import EmbeddingError
from app.services.llm.errors import classify_error


def _caused(exc: BaseException, cause: BaseException) -> BaseException:
    """`exc` raised `from cause`, as the wrappers raise it."""
    try:
        raise exc from cause
    except BaseException as raised:
        return raised


def _boto(code: str, message: str = "", status: int | None = None) -> BotoClientError:
    response: dict = {"Error": {"Code": code, "Message": message}}
    if status is not None:
        response["ResponseMetadata"] = {"HTTPStatusCode": status}
    return BotoClientError(response, "Converse")


def _genai(code: int, status: str, message: str) -> GenaiClientError:
    return GenaiClientError(code, {"error": {"code": code, "message": message, "status": status}})


_OPENAI_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/embeddings")


def _openai(cls: type[openai.APIStatusError], status: int, message: str, code: str | None = None):
    body = {"error": {"message": message, "code": code}}
    response = httpx2.Response(status, request=_OPENAI_REQUEST)
    return cls(message, response=response, body=body)


def _validation_error() -> ValidationError:
    class Answer(BaseModel):
        answer: str

    try:
        Answer.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


# Vertex's refusal for a GCP project without billing, as the API logged it.
_VERTEX_BILLING = _caused(
    GooglePermissionDeniedError("Error calling model 'gemini-3.8-flash' (Forbidden)"),
    _genai(403, "PERMISSION_DENIED", "This API method requires billing to be enabled."),
)


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        pytest.param(_VERTEX_BILLING, "billing", id="vertex-403-billing"),
        pytest.param(
            GoogleRateLimitError("Resource exhausted. Please try again later."),
            "rate_limit",
            id="vertex-429",
        ),
        pytest.param(
            _genai(429, "RESOURCE_EXHAUSTED", "Quota exceeded for generate_content per minute."),
            "rate_limit",
            id="genai-429-per-minute-quota-is-not-billing",
        ),
        pytest.param(
            _genai(404, "NOT_FOUND", "Model not found."), "invalid_request", id="genai-404"
        ),
        pytest.param(
            GenaiServerError(503, {"error": {"code": 503, "message": "Unavailable"}}),
            "unavailable",
            id="genai-503",
        ),
        pytest.param(
            _boto("ThrottlingException", "Too many requests", 429),
            "rate_limit",
            id="bedrock-throttle",
        ),
        pytest.param(
            _boto("AccessDeniedException", "You don't have access to the model.", 403),
            "auth",
            id="bedrock-access-denied",
        ),
        pytest.param(
            _boto(
                "AccessDeniedException",
                "Model access is denied due to INVALID_PAYMENT_INSTRUMENT: "
                "A valid payment instrument must be provided.",
                403,
            ),
            "billing",
            id="bedrock-payment",
        ),
        pytest.param(
            _boto("modelStreamErrorException", "stream broke"),
            "unavailable",
            id="bedrock-stream-event-without-status",
        ),
        pytest.param(
            _boto("ValidationException", "The model identifier is invalid.", 400),
            "invalid_request",
            id="bedrock-validation",
        ),
        pytest.param(
            ReadTimeoutError(endpoint_url="https://bedrock-runtime"),
            "timeout",
            id="boto-read-timeout",
        ),
        pytest.param(
            EndpointConnectionError(endpoint_url="https://bedrock-runtime"),
            "unavailable",
            id="boto-no-connection",
        ),
        pytest.param(NoCredentialsError(), "auth", id="boto-no-credentials"),
        pytest.param(
            _openai(
                openai.RateLimitError,
                429,
                "You exceeded your current quota, please check your plan and billing details.",
                "insufficient_quota",
            ),
            "billing",
            id="openai-insufficient-quota",
        ),
        pytest.param(
            _openai(
                openai.RateLimitError,
                429,
                "Rate limit reached for requests.",
                "rate_limit_exceeded",
            ),
            "rate_limit",
            id="openai-429",
        ),
        pytest.param(
            _caused(
                EmbeddingError("Failed to generate embeddings"),
                _openai(openai.AuthenticationError, 401, "Incorrect API key provided."),
            ),
            "auth",
            id="embedding-error-wrapping-openai-401",
        ),
        pytest.param(
            openai.APITimeoutError(request=_OPENAI_REQUEST), "timeout", id="openai-timeout"
        ),
        pytest.param(
            openai.APIConnectionError(request=_OPENAI_REQUEST),
            "unavailable",
            id="openai-connection",
        ),
        pytest.param(TimeoutError(), "timeout", id="timeout"),
        pytest.param(
            _caused(
                OperationalError("SELECT 1", {}, ConnectionRefusedError()),
                ConnectionRefusedError(),
            ),
            "unavailable",
            id="database-down",
        ),
        pytest.param(_validation_error(), "bad_response", id="pydantic"),
        pytest.param(
            ValueError("bedrock_structured_output_parse_failed: not JSON"),
            "bad_response",
            id="structured-parse",
        ),
        pytest.param(OutputParserException("bad"), "bad_response", id="output-parser"),
        pytest.param(RuntimeError("provider exploded"), "unknown", id="unknown"),
    ],
)
def test_classify_error(exc: BaseException, kind: str):
    assert classify_error(exc) == kind


def test_a_cycle_in_the_chain_ends():
    a, b = RuntimeError("a"), RuntimeError("b")
    a.__cause__, b.__cause__ = b, a
    assert classify_error(a) == "unknown"
