"""Responses/OAuth adapter regression tests, without network calls."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletionMessage

from research_agent.api_settings import APISettings
from research_agent.openai_oauth import OAuthError
from research_agent.openai_oauth_completer import (
    InvalidAssistantMessage, OpenAIOAuthTurnCompleterConfig, TOOL_NAMESPACE,
)


def config(tmp_path, **kwargs):
    return OpenAIOAuthTurnCompleterConfig(credentials_dir=str(tmp_path / "oauth"), **kwargs)


def completed():
    return SimpleNamespace(output=[SimpleNamespace(type="message", content=[
        SimpleNamespace(type="output_text", text="OK")])], usage=None)


def completed_with_content(content):
    return SimpleNamespace(
        id="resp-offline",
        output=[SimpleNamespace(type="message", content=[content])],
        usage=None,
    )


@pytest.mark.asyncio
async def test_invalid_message_retries_twice_with_same_conversation(tmp_path):
    c = config(tmp_path).build()
    c._complete_with_retries = AsyncMock(side_effect=[
        completed_with_content(SimpleNamespace(type="output_text", text="")),
        completed_with_content(SimpleNamespace(type="output_text", text="")),
        completed(),
    ])
    conversation = [{"role": "user", "content": "continue"}]
    result = await c.async_completion(conversation)
    assert result.output_messages[0].content == "OK"
    assert c._complete_with_retries.call_count == 3
    assert conversation == [{"role": "user", "content": "continue"}]
    assert len({str(call.args[0]) for call in c._complete_with_retries.call_args_list}) == 1
    assert c._call_items == {}


@pytest.mark.asyncio
async def test_invalid_message_retry_limit_and_refusal(tmp_path):
    c = config(tmp_path).build()
    c._complete_with_retries = AsyncMock(return_value=completed_with_content(
        SimpleNamespace(type="output_text", text=""),
    ))
    with pytest.raises(InvalidAssistantMessage, match="after 3 completed responses"):
        await c.async_completion([{"role": "user", "content": "continue"}])
    assert c._complete_with_retries.call_count == 3
    c._complete_with_retries = AsyncMock(return_value=completed_with_content(
        SimpleNamespace(type="refusal", refusal="Cannot assist"),
    ))
    result = await c.async_completion([{"role": "user", "content": "continue"}])
    assert result.output_messages[0].refusal == "Cannot assist"
    assert c._complete_with_retries.call_count == 1


@pytest.mark.asyncio
async def test_reasoning_only_completed_response_is_retried(tmp_path):
    c = config(tmp_path).build()
    reasoning_only = SimpleNamespace(id="resp-offline", output=[
        SimpleNamespace(type="reasoning", encrypted_content=None),
    ], usage=None)
    c._complete_with_retries = AsyncMock(side_effect=[reasoning_only, completed()])
    result = await c.async_completion([{"role": "user", "content": "continue"}])
    assert result.output_messages[0].content == "OK"
    assert c._complete_with_retries.call_count == 2


def test_roles_and_model_selection(tmp_path):
    original = config(tmp_path, model="gpt-6-astra", proxy="http://localhost:7890")
    settings = APISettings.from_config(original)
    planner = settings.new_config(basicagent_tools=[])
    checker = settings.new_config(basicagent_tools=[])
    assert planner is not checker
    assert planner.basicagent_tools is not checker.basicagent_tools
    assert planner.credentials_dir == original.credentials_dir
    assert planner.proxy == original.proxy
    assert planner.model == "gpt-6-astra"
    assert config(tmp_path).model == "gpt-6-luna"
    assert config(tmp_path, model="gpt6luna").build().model == "gpt-6-luna"


def test_runner_model_default_and_override(monkeypatch):
    import sys
    from research_agent.runner import parse_args
    required = ["runner", "--container", "test", "--workspace", "/tmp", "--run-dir", "/tmp/test",
                "--run-id", "test", "--run-group-id", "test", "--api-config", "openai-oauth",
                "--skill-target", "all-in-one"]
    monkeypatch.setattr(sys, "argv", required)
    assert parse_args().model == "gpt-6-luna"
    monkeypatch.setattr(sys, "argv", required + ["--model", "gpt-6-astra"])
    assert parse_args().model == "gpt-6-astra"
    monkeypatch.setattr(sys, "argv", required + ["--reasoning-effort", "medium"])
    assert parse_args().reasoning_effort == "medium"
    monkeypatch.setenv("RESEARCH_AGENT_REASONING_EFFORT", "medium")
    monkeypatch.setattr(sys, "argv", required)
    assert parse_args().reasoning_effort == "medium"


def test_protocol_conversion_and_fields(tmp_path):
    c = config(tmp_path).build()
    c.tools = [{"type": "function", "name": "submit_subtask", "parameters": {"type": "object"}}]
    history = [{"role": "system", "content": "unchanged system prompt"},
               {"role": "user", "content": "unchanged plan"},
               {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function",
                  "function": {"name": "submit_subtask", "arguments": "{}"}}]},
               {"role": "tool", "tool_call_id": "call-1", "content": "rejected"}]
    body = c._request_body(history)
    assert history[0]["role"] == "system"
    assert body["input"][0]["role"] == "developer"
    assert body["input"][0]["content"] == history[0]["content"]
    assert body["input"][2]["namespace"] == TOOL_NAMESPACE
    assert body["input"][3]["type"] == "function_call_output"
    assert body["input"][3]["call_id"] == "call-1"
    assert body["store"] is False and body["stream"] is True
    assert body["tools"][0]["type"] == "namespace"
    for forbidden in ("max_output_tokens", "temperature", "top_p", "previous_response_id"):
        assert forbidden not in body


@pytest.mark.asyncio
async def test_each_request_reads_credentials(tmp_path):
    c = config(tmp_path).build()
    c._token = AsyncMock(side_effect=[{"access_token": "first"}, {"access_token": "rotated"}])
    c._stream_response = AsyncMock(return_value=completed())
    for _ in range(2):
        output = await c.async_completion([{"role": "user", "content": "OK"}])
        assert output.output_messages[0].content == "OK"
    assert c._stream_response.call_args_list[0].args[1]["access_token"] == "first"
    assert c._stream_response.call_args_list[1].args[1]["access_token"] == "rotated"


@pytest.mark.asyncio
async def test_401_refresh_once(tmp_path):
    c = config(tmp_path).build()
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    error = openai.AuthenticationError("expired", response=httpx.Response(401, request=request), body={})
    c._token = AsyncMock(side_effect=[{"access_token": "old"}, {"access_token": "new"}])
    c._stream_response = AsyncMock(side_effect=[error, completed()])
    await c.async_completion([{"role": "user", "content": "OK"}])
    assert c._token.call_args_list[1].kwargs == {"force": True, "failed_access_token": "old"}
    c._token = AsyncMock(return_value={"access_token": "still-invalid"})
    c._stream_response = AsyncMock(side_effect=error)
    with pytest.raises(openai.AuthenticationError):
        await c.async_completion([{"role": "user", "content": "OK"}])
    assert c._stream_response.call_count == 2


def test_quota_failure_is_not_transient(tmp_path):
    c = config(tmp_path).build()
    assert isinstance(c._failure("subscription_sharing_usage_limit_exceeded"), OAuthError)


class MockStream:
    def __init__(self, events):
        self.events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def __aiter__(self):
        for event in self.events:
            yield event


@pytest.mark.asyncio
async def test_empty_terminal_output_reconstruction(tmp_path):
    from openai.types.responses import Response
    c = config(tmp_path).build()
    response = Response.model_construct(output=[], usage=None)
    events = [SimpleNamespace(type="response.output_item.done", output_index=0, item=completed().output[0]),
              SimpleNamespace(type="response.completed", response=response)]
    c.__dict__["_client"] = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=MockStream(events))))
    result = await c._stream_response({}, {"access_token": "dummy"})
    assert result.output[0].content[0].text == "OK"
    # Done items alone must never be treated as successful inference.
    c._client.responses.create = AsyncMock(return_value=MockStream(events[:1]))
    with pytest.raises(openai.APIConnectionError):
        await c._stream_response({}, {"access_token": "dummy"})


def test_reasoning_replay_tracks_context_units(tmp_path):
    c = config(tmp_path).build()
    c._call_items["keep"] = {"type": "function_call", "call_id": "keep", "name": "inspect", "arguments": "{}"}
    c._reasoning_items["keep"] = [{"id": "r1", "type": "reasoning", "summary": [], "encrypted_content": "opaque"}]
    c._call_items["trimmed"] = {"type": "function_call"}
    message = ChatCompletionMessage(role="assistant", tool_calls=[{"id": "keep", "type": "function",
                "function": {"name": "inspect", "arguments": "{}"}}]).model_dump(exclude_none=True)
    body = c._request_body([message])
    assert body["input"][0]["type"] == "reasoning"
    assert "trimmed" not in c._call_items
