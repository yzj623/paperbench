"""Fault injection at stream/request/refresh boundaries; never calls an API."""
import asyncio
from email.utils import formatdate
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpcore
import httpx
import openai
import pytest
from openai.types.responses import Response

from research_agent.openai_oauth import OAuthClient, OAuthError, OAuthHTTPError, PLAN_SCOPES, private_json, read_json
from research_agent.openai_oauth_completer import OpenAIOAuthTurnCompleterConfig
from research_agent.openai_oauth_resilience import (
    OpenAIOAuthRetryConfig, OpenAIResponseFailure, is_retryable, retry_after,
)


def http_error(status, code=None, headers=None):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status, request=request, headers=headers or {})
    error_type = {400: openai.BadRequestError, 401: openai.AuthenticationError, 403: openai.PermissionDeniedError,
                  404: openai.NotFoundError, 409: openai.ConflictError, 422: openai.UnprocessableEntityError,
                  429: openai.RateLimitError}.get(status, openai.InternalServerError if status >= 500 else openai.APIStatusError)
    return error_type("fault injection", response=response, body={"code": code})


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504, 599])
def test_retryable_http_status(status):
    assert is_retryable(http_error(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_http_status(status):
    assert not is_retryable(http_error(status))


@pytest.mark.parametrize("code", ["insufficient_quota", "credit_balance_exhausted", "organization_spend_limit_exceeded",
                                   "project_spend_limit_exceeded", "organization_usage_limit_exceeded",
                                   "subscription_sharing_usage_limit_exceeded", "misalignment_policy_violation"])
def test_quota_policy_errors_never_retry(code):
    assert not is_retryable(http_error(429, code))
    assert not is_retryable(OpenAIResponseFailure(code, status=200))


@pytest.mark.parametrize("code", ["server_error", "rate_limit_exceeded", "slow_down", "server_is_overloaded",
                                   "subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable"])
def test_sse_and_sdk_generic_error_codes(code):
    assert is_retryable(OpenAIResponseFailure(code, status=200))
    assert is_retryable(openai.APIError("stream failure", request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
                                      body={"error": {"code": code}}))


@pytest.mark.parametrize("error", [httpx.RemoteProtocolError("incomplete chunked read"),
                                    httpx.ReadError("reset"), httpx.ConnectError("connect"),
                                    httpx.ReadTimeout("timeout"), httpx.PoolTimeout("pool"),
                                    httpx.WriteError("write"), httpx.ProxyError("proxy"),
                                    httpcore.RemoteProtocolError("chunked read"),
                                    ConnectionResetError("reset"), TimeoutError("timeout")])
def test_transport_taxonomy(error):
    assert is_retryable(error)


def test_cancellation_and_local_bug_are_not_retried():
    for exc in [asyncio.CancelledError(), ValueError("bug"), httpx.LocalProtocolError("bad request"),
                httpx.UnsupportedProtocol("bad URL"), httpcore.LocalProtocolError("bad request")]:
        assert not is_retryable(exc)


def test_retry_after_formats_and_server_override():
    assert retry_after(http_error(429, headers={"Retry-After": "4.5"})) == 4.5
    assert retry_after(http_error(503, headers={"Retry-After-Ms": "1500"})) == 1.5
    assert retry_after(http_error(503, headers={"Retry-After": formatdate(1060, usegmt=True)}), now=1000) == 60
    assert retry_after(http_error(503, headers={"Retry-After": "NaN"})) is None
    assert not is_retryable(http_error(503, headers={"x-should-retry": "false"}))


def test_wait_respects_server_minimum():
    retrying = OpenAIOAuthRetryConfig(wait_min=0, wait_max=10).build()
    exc = http_error(429, headers={"Retry-After": "4.5"})
    state = SimpleNamespace(attempt_number=1, outcome=SimpleNamespace(exception=lambda: exc))
    assert 4.5 <= retrying.wait(state) <= 10


def completer(tmp_path, *, attempts=3, deadline=5, wait_max=0):
    c = OpenAIOAuthTurnCompleterConfig(credentials_dir=str(tmp_path / "credentials"),
        retry_config=OpenAIOAuthRetryConfig(wait_min=0, wait_max=wait_max, stop_after=deadline, max_attempts=attempts)).build()
    c._token = AsyncMock(return_value={"access_token": "test-bearer-never-log"})
    return c


class FaultStream:
    def __init__(self, events=(), failure=None, close_failure=None):
        self.events, self.failure, self.close_failure = events, failure, close_failure
        self.response = SimpleNamespace(headers={"x-request-id": "req-fault-test"})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        if self.close_failure:
            raise self.close_failure

    async def __aiter__(self):
        for event in self.events:
            yield event
        if self.failure:
            raise self.failure


def success_stream():
    message = SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="OK")])
    return FaultStream([SimpleNamespace(type="response.output_item.done", output_index=0, item=message),
                        SimpleNamespace(type="response.completed", response=Response.model_construct(output=[], usage=None))])


def install_streams(c, *streams):
    create = AsyncMock(side_effect=list(streams))
    c.__dict__["_client"] = SimpleNamespace(responses=SimpleNamespace(create=create))
    return create


@pytest.mark.asyncio
async def test_actual_crash_reproduced_and_recovered_without_partial_tools(tmp_path):
    c = completer(tmp_path)
    partial = SimpleNamespace(type="function_call", call_id="must-not-execute", name="bash", arguments='{"cmd":"partial"}')
    broken = FaultStream([SimpleNamespace(type="response.output_item.done", output_index=0, item=partial)],
                         failure=httpx.RemoteProtocolError("peer closed connection without sending complete message body"))
    create = install_streams(c, broken, success_stream())
    result = await c.async_completion([{"role": "user", "content": "test"}])
    assert result.output_messages[0].content == "OK"
    assert not result.output_messages[0].tool_calls
    assert "must-not-execute" not in c._call_items
    assert create.call_count == 2
    assert create.call_args_list[0].kwargs["input"] == create.call_args_list[1].kwargs["input"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [httpx.ReadTimeout("timeout"), httpcore.RemoteProtocolError("reset"),
                                    openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))])
async def test_stream_transport_recovers(tmp_path, failure):
    c = completer(tmp_path)
    create = install_streams(c, FaultStream(failure=failure), success_stream())
    assert (await c.async_completion([{"role": "user", "content": "test"}])).output_messages[0].content == "OK"
    assert create.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
async def test_request_status_recovers_at_create_boundary(tmp_path, status):
    c = completer(tmp_path)
    create = install_streams(c, http_error(status), success_stream())
    assert (await c.async_completion([{"role": "user", "content": "test"}])).output_messages[0].content == "OK"
    assert create.call_count == 2


@pytest.mark.asyncio
async def test_subscription_limit_waits_and_retries_same_request(tmp_path, monkeypatch):
    c = completer(tmp_path, attempts=1, deadline=5)
    limit_code = "subscription_sharing_usage_limit_exceeded"
    http_limit = http_error(429, limit_code, headers={"Retry-After": "601"})
    stream_limit = FaultStream([SimpleNamespace(type="error", code=limit_code, param=None)])
    create = install_streams(c, http_limit, stream_limit, success_stream())
    sleep = AsyncMock()
    monkeypatch.setattr("research_agent.openai_oauth_completer.asyncio.sleep", sleep)

    result = await c.async_completion([{"role": "user", "content": "test"}])

    assert result.output_messages[0].content == "OK"
    assert [call.args[0] for call in sleep.call_args_list] == [601.0, 300.0]
    assert create.call_count == 3
    assert create.call_args_list[0].kwargs["input"] == create.call_args_list[2].kwargs["input"]


@pytest.mark.asyncio
async def test_explicit_stream_error_and_incomplete_are_distinguished(tmp_path):
    c = completer(tmp_path)
    event = SimpleNamespace(type="error", code="server_is_overloaded", param=None)
    create = install_streams(c, FaultStream([event]), success_stream())
    assert (await c.async_completion([{"role": "user", "content": "test"}])).output_messages[0].content == "OK"
    assert create.call_count == 2
    event = SimpleNamespace(type="response.incomplete", response=SimpleNamespace(
        incomplete_details=SimpleNamespace(reason="max_output_tokens")))
    create = install_streams(c, FaultStream([event]))
    with pytest.raises(OpenAIResponseFailure, match="response.incomplete"):
        await c.async_completion([{"role": "user", "content": "test"}])
    assert create.call_count == 1


@pytest.mark.asyncio
async def test_sse_and_sdk_sse_error_events_recover(tmp_path):
    c = completer(tmp_path)
    failed = SimpleNamespace(type="response.failed", response=SimpleNamespace(error=SimpleNamespace(code="subscription_sharing_usage_unavailable")))
    generic = openai.APIError("temporary stream error", request=httpx.Request("POST", "https://api.openai.com/v1/responses"), body={"code": "server_error"})
    create = install_streams(c, FaultStream([failed]), FaultStream(failure=generic), success_stream())
    result = await c.async_completion([{"role": "user", "content": "test"}])
    assert result.output_messages[0].content == "OK" and create.call_count == 3


@pytest.mark.asyncio
async def test_early_eof_retries_and_completed_cleanup_does_not(tmp_path):
    c = completer(tmp_path)
    create = install_streams(c, FaultStream(), success_stream())
    await c.async_completion([{"role": "user", "content": "test"}])
    assert create.call_count == 2
    success = success_stream()
    success.close_failure = httpx.RemoteProtocolError("close failed after completion")
    create = install_streams(c, success)
    await c.async_completion([{"role": "user", "content": "test"}])
    assert create.call_count == 1
    success = success_stream()
    success.close_failure = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    create = install_streams(c, success)
    await c.async_completion([{"role": "user", "content": "test"}])
    assert create.call_count == 1


@pytest.mark.asyncio
async def test_terminal_http_diagnostic_does_not_echo_credentials(tmp_path):
    c = completer(tmp_path)
    exc = http_error(403, "permission_denied")
    exc.message = "server echoed test-bearer-never-log"
    create = install_streams(c, exc)
    with pytest.raises(openai.PermissionDeniedError) as caught:
        await c.async_completion([{"role": "user", "content": "test"}])
    assert create.call_count == 1
    assert "test-bearer-never-log" not in str(caught.value)
    assert caught.value.status_code == 403


@pytest.mark.asyncio
async def test_retry_count_limit_and_no_credential_logging(tmp_path, caplog):
    c = completer(tmp_path)
    c._stream_response = AsyncMock(side_effect=httpx.RemoteProtocolError("broken"))
    with pytest.raises(httpx.RemoteProtocolError):
        await c.async_completion([{"role": "user", "content": "test"}])
    assert c._stream_response.call_count == 3
    assert "test-bearer-never-log" not in caplog.text
    assert "OpenAI retry" in caplog.text


@pytest.mark.asyncio
async def test_deadline_and_external_cancel(tmp_path):
    c = completer(tmp_path, deadline=.02)
    async def blocked(*_):
        await asyncio.sleep(10)
    c._stream_response = blocked
    with pytest.raises(OAuthError, match="deadline exhausted"):
        await c.async_completion([{"role": "user", "content": "test"}])
    c = completer(tmp_path)
    c._stream_response = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await c.async_completion([{"role": "user", "content": "test"}])
    assert c._stream_response.call_count == 1


@pytest.mark.asyncio
async def test_long_retry_after_is_deferred_not_ignored(tmp_path):
    c = completer(tmp_path, wait_max=1)
    c._stream_response = AsyncMock(side_effect=http_error(503, headers={"Retry-After": "60"}))
    with pytest.raises(openai.InternalServerError):
        await c.async_completion([{"role": "user", "content": "test"}])
    assert c._stream_response.call_count == 1


@pytest.mark.asyncio
async def test_refresh_transient_failure_and_401_do_not_loop(tmp_path):
    c = completer(tmp_path)
    c._token = AsyncMock(side_effect=[{"access_token": "old"},
        OAuthHTTPError("HTTP 503", status_code=503, code="server_error"), {"access_token": "new"}])
    successful = SimpleNamespace(output=[success_stream().events[0].item], usage=None)
    c._stream_response = AsyncMock(side_effect=[http_error(401), successful])
    assert (await c.async_completion([{"role": "user", "content": "test"}])).output_messages[0].content == "OK"
    assert c._token.call_args_list[2].kwargs == {"force": True, "failed_access_token": "old"}


def test_invalid_refresh_clears_only_unusable_tokens(tmp_path, monkeypatch):
    c = OAuthClient(tmp_path / "credentials")
    record = {"client_id": "saved-client", "subject": "saved-account", "access_token": "invalid-access",
              "refresh_token": "invalid-refresh", "scopes": sorted(PLAN_SCOPES), "expires_at": 0}
    private_json(c.credentials, record)
    def invalid(_):
        raise OAuthHTTPError("expired refresh", status_code=400, code="invalid_grant")
    monkeypatch.setattr(c, "token_exchange", invalid)
    with pytest.raises(OAuthHTTPError):
        c.refresh()
    saved = read_json(c.credentials)
    assert "access_token" not in saved and "refresh_token" not in saved
    assert saved["client_id"] == "saved-client" and saved["subject"] == "saved-account"
    assert saved["reauthorization_required"] is True
