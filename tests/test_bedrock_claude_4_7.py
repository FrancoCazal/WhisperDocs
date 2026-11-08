"""Claude 4.7+ on Bedrock through Converse (audit N7).

AWS serves Claude Sonnet 5.5 through Converse (`global.` and `us.` inference
profiles). It rejects a non-default temperature and a forced tool choice,
thinks adaptively, and counts that thinking toward `max_tokens`. These tests
run the real BedrockProvider and ChatBedrockConverse against botocore's
Stubber: the request langchain-aws builds is checked, and a canned response
comes back. Nothing reaches AWS.
"""

from __future__ import annotations

import json

import pytest
from botocore.stub import Stubber
from pydantic import BaseModel, Field

from app.agent.schemas import RAGAnswer
from app.config import settings
from app.services.llm.bedrock_provider import BedrockProvider, _output_schema
from app.services.llm.errors import classify_error
from app.services.llm.model_policy import THINKING_HEADROOM_TOKENS

SONNET_5_5 = "global.anthropic.claude-sonnet-5-5"
SONNET_4_5 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

_ANSWER = {
    "answer": "Vacations are 14 days [1].",
    "uses_context": True,
    "confidence": 0.9,
    "cited_chunks": [1],
    "refused": False,
    "refuse_reason": None,
}


def _converse_response(content: list[dict], stop_reason: str = "end_turn") -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": 12, "outputTokens": 5, "totalTokens": 17},
        "metrics": {"latencyMs": 10},
    }


# Adaptive thinking comes back as a reasoning block before the text; its text
# is empty under Claude Sonnet 5.5's default display.
_THINKING = {"reasoningContent": {"reasoningText": {"text": "", "signature": "sig"}}}


@pytest.fixture
def bedrock(monkeypatch):
    """A BedrockProvider whose botocore client answers from a script. Yields
    (provider, stubber, requests), where `requests` collects each Converse
    request's parameters."""
    monkeypatch.setattr(settings, "aws_access_key_id", "test")
    monkeypatch.setattr(settings, "aws_secret_access_key", "test")
    monkeypatch.setattr(settings, "aws_region", "us-east-1")
    monkeypatch.setattr(settings, "bedrock_claude_effort", "low")
    provider = BedrockProvider()
    requests: list[dict] = []
    provider._client.meta.events.register(
        "before-parameter-build.bedrock-runtime.Converse",
        lambda params, **_kw: requests.append(params),
    )
    with Stubber(provider._client) as stubber:
        yield provider, stubber, requests
        stubber.assert_no_pending_responses()


async def test_claude_4_7_gets_effort_and_thinking_room_but_no_temperature(bedrock):
    provider, stubber, requests = bedrock
    stubber.add_response("converse", _converse_response([_THINKING, {"text": "Hello"}]))

    response = await provider.generate(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        model=SONNET_5_5,
        temperature=0.3,
        max_tokens=100,
    )

    (request,) = requests
    assert request["modelId"] == SONNET_5_5
    assert request["inferenceConfig"] == {"maxTokens": 100 + THINKING_HEADROOM_TOKENS}
    assert request["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
    }
    assert "toolConfig" not in request
    assert response.text == "Hello"
    assert (response.tokens_input, response.tokens_output) == (12, 5)


async def test_without_an_effort_the_model_default_applies(bedrock, monkeypatch):
    provider, stubber, requests = bedrock
    monkeypatch.setattr(settings, "bedrock_claude_effort", None)
    stubber.add_response("converse", _converse_response([{"text": "Hello"}]))

    await provider.generate(system="s", messages=[], model=SONNET_5_5, max_tokens=100)

    (request,) = requests
    assert "additionalModelRequestFields" not in request
    assert request["inferenceConfig"] == {"maxTokens": 100 + THINKING_HEADROOM_TOKENS}


async def test_claude_4_5_requests_are_unchanged(bedrock):
    provider, stubber, requests = bedrock
    stubber.add_response("converse", _converse_response([{"text": "Hello"}]))

    await provider.generate(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        model=SONNET_4_5,
        temperature=0.3,
        max_tokens=100,
    )

    (request,) = requests
    assert request["inferenceConfig"] == {"maxTokens": 100, "temperature": 0.3}
    assert "additionalModelRequestFields" not in request


async def test_claude_4_7_structured_output_is_a_json_schema(bedrock):
    provider, stubber, requests = bedrock
    stubber.add_response("converse", _converse_response([_THINKING, {"text": json.dumps(_ANSWER)}]))

    parsed, response = await provider.generate_structured(
        system="s",
        messages=[{"role": "user", "content": "q"}],
        model=SONNET_5_5,
        response_model=RAGAnswer,
        max_tokens=1000,
    )

    (request,) = requests
    assert "toolConfig" not in request  # no forced tool call
    text_format = request["outputConfig"]["textFormat"]
    assert text_format["type"] == "json_schema"
    schema = json.loads(text_format["structure"]["jsonSchema"]["schema"])
    confidence = schema["properties"]["confidence"]
    assert "minimum" not in confidence and "maximum" not in confidence
    assert schema["additionalProperties"] is False
    assert parsed == RAGAnswer(**_ANSWER)
    assert response.tokens_input == 12


async def test_a_bound_the_schema_could_not_carry_is_still_checked(bedrock):
    provider, stubber, _requests = bedrock
    out_of_bounds = {**_ANSWER, "confidence": 1.5}
    stubber.add_response("converse", _converse_response([{"text": json.dumps(out_of_bounds)}]))

    with pytest.raises(ValueError, match="structured_output_parse_failed") as caught:
        await provider.generate_structured(
            system="s",
            messages=[{"role": "user", "content": "q"}],
            model=SONNET_5_5,
            response_model=RAGAnswer,
        )

    assert classify_error(caught.value) == "bad_response"


async def test_claude_4_5_structured_output_still_forces_the_tool(bedrock):
    provider, stubber, requests = bedrock
    tool_use = {"toolUse": {"toolUseId": "t1", "name": "RAGAnswer", "input": _ANSWER}}
    stubber.add_response("converse", _converse_response([tool_use], stop_reason="tool_use"))

    parsed, _response = await provider.generate_structured(
        system="s",
        messages=[{"role": "user", "content": "q"}],
        model=SONNET_4_5,
        response_model=RAGAnswer,
    )

    (request,) = requests
    assert request["toolConfig"]["toolChoice"] == {"tool": {"name": "RAGAnswer"}}
    assert "outputConfig" not in request
    assert parsed == RAGAnswer(**_ANSWER)


class _Inner(BaseModel):
    tags: list[str] = Field(max_length=3)


class _Constrained(BaseModel):
    minimum: int = Field(ge=1, le=10, description="A field named like a keyword")
    name: str = Field(min_length=2)
    inner: _Inner


def test_output_schema_moves_constraints_into_descriptions():
    schema = _output_schema(_Constrained)

    props = schema["properties"]
    assert set(props) == {"minimum", "name", "inner"}  # the property survives
    assert "maximum" not in props["minimum"]
    assert props["minimum"]["description"].endswith("{minimum: 1, maximum: 10}")
    assert "minLength" not in props["name"]
    tags = schema["$defs"]["_Inner"]["properties"]["tags"]
    assert "maxItems" not in tags and "{maxItems: 3}" in tags["description"]
    # Pydantic still holds the bounds.
    with pytest.raises(ValueError):
        _Constrained(minimum=11, name="ab", inner=_Inner(tags=[]))
