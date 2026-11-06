"""ChatGPT-plan Responses adapter; RA's planning/execution/review protocol is unchanged."""
from __future__ import annotations

import asyncio
import functools
import logging
import os
from pathlib import Path
from typing import Any, Literal, Unpack

import httpx
import httpcore
import openai
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessage
from openai.types.responses import ResponseUsage
from paperbench.solvers.basicagent.api import is_chat_completion_message_param_list
from paperbench.solvers.basicagent.completer import BasicAgentTurnCompleterConfig
from preparedness_turn_completer.oai_responses_turn_completer.converters import convert_conversation_to_response_input
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.type_helpers import chat_completion_message_to_message_param
from preparedness_turn_completer.utils import RetryConfig, warn_about_non_empty_params
from pydantic import ConfigDict, Field

from .openai_oauth import OAuthClient, OAuthError, OAuthHTTPError, default_credentials_dir as oauth_credentials_dir
from .openai_oauth_resilience import (
    OpenAIOAuthRetryConfig, OpenAIResponseFailure, error_fields, is_account_usage_limit,
    is_retryable, oauth_retry_config, retry_after,
)

DEFAULT_MODEL = "gpt-6-luna"
TOOL_NAMESPACE = "research_agent"
MAX_INVALID_MESSAGE_RETRIES = 2
logger = logging.getLogger(__name__)


class InvalidAssistantMessage(OAuthError):
    """A completed API response cannot be passed to RA as an assistant message."""


def normalize_model(model: str) -> str:
    return DEFAULT_MODEL if model in {"gpt6luna", "gpt6-luna"} else model


def default_credentials_dir() -> str:
    return str(oauth_credentials_dir())


def default_proxy() -> str | None:
    return os.environ.get("RESEARCH_AGENT_OPENAI_PROXY") or None


class OpenAIOAuthTurnCompleter(TurnCompleter):
    def __init__(self, *, model=DEFAULT_MODEL, credentials_dir=None, proxy=None,
                 context_window=128_000, reasoning=NOT_GIVEN, tools=None, retry_config=None,
                 wait_for_account_usage_limit=True):
        self.model = normalize_model(model)
        self.context_window = self.n_ctx = context_window
        self.encoding_name = "o200k_base"
        self.reasoning = reasoning
        self.tools = list(tools or [])
        self.retry_config = oauth_retry_config(retry_config)
        self.wait_for_account_usage_limit = wait_for_account_usage_limit
        self.proxy = proxy
        self.oauth = OAuthClient(Path(credentials_dir or default_credentials_dir()), proxy)
        # Raw Responses call/reasoning items are retained only while the matching
        # tool-call unit remains in RA's context. They never include credentials.
        self._call_items: dict[str, dict] = {}
        self._reasoning_items: dict[str, list[dict]] = {}

    class Completion(TurnCompleter.Completion):
        usage: ResponseUsage | None = None

    @functools.cached_property
    def _client(self):
        options = {"http_client": httpx.AsyncClient(proxy=self.proxy, timeout=600)} if self.proxy else {}
        # Real bearer credentials are supplied per request, never cached here.
        return openai.AsyncOpenAI(api_key="oauth-per-request", base_url="https://api.openai.com/v1",
                                  max_retries=0, timeout=600, **options)

    def completion(self, conversation, **params):
        raise NotImplementedError("Use async_completion")

    def _request_body(self, conversation):
        adjusted = [dict(message, role="developer") if message["role"] == "system"
                    else dict(message) for message in conversation]
        for message in adjusted:
            if message["role"] == "assistant":
                message.setdefault("content", None)
        items = convert_conversation_to_response_input(adjusted)
        active_ids = {item["call_id"] for item in items if item["type"] == "function_call"}
        self._call_items = {key: value for key, value in self._call_items.items() if key in active_ids}
        self._reasoning_items = {key: value for key, value in self._reasoning_items.items() if key in active_ids}
        replay, seen_reasoning = [], set()
        for item in items:
            if item["type"] == "function_call":
                call_id = item["call_id"]
                for reasoning in self._reasoning_items.get(call_id, []):
                    if reasoning["id"] not in seen_reasoning:
                        replay.append(reasoning)
                        seen_reasoning.add(reasoning["id"])
                item = dict(self._call_items.get(call_id, item))
                item["namespace"] = TOOL_NAMESPACE
            replay.append(item)
        body = {"model": self.model, "input": replay, "store": False, "stream": True,
                "include": ["reasoning.encrypted_content"]}
        if self.tools:
            body["tools"] = [{"type": "namespace", "name": TOOL_NAMESPACE,
                              "description": "ResearchAgent's local protocol and workspace tools.",
                              "tools": self.tools}]
        if self.reasoning is not None and not isinstance(self.reasoning, NotGiven):
            body["reasoning"] = (self.reasoning.model_dump(exclude_none=True)
                                 if hasattr(self.reasoning, "model_dump") else self.reasoning)
        return body

    @staticmethod
    def _failure(code, *, status=None, param=None, request_id=None, headers=None, body=None):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(status or 400, request=request)
        body = body or {"error": {"code": code, "param": param}}
        if code in {"context_length_exceeded", "input_context_length_exceeded"}:
            return openai.BadRequestError("input exceeds the context window", response=response, body=body)
        return OpenAIResponseFailure(code, status=status, param=param, request_id=request_id,
                                     headers=headers, body=body)

    async def _token(self, **kwargs):
        try:
            return await asyncio.to_thread(self.oauth.refresh, **kwargs)
        except OAuthHTTPError as exc:
            # The OAuth-specific retry policy handles HTTP status, exact code,
            # Retry-After and terminal refresh-token failures without losing metadata.
            raise
        except OAuthError as exc:
            if str(exc).startswith("Network error:"):
                raise openai.APIConnectionError(message="OpenAI OAuth refresh network error; saved credentials retained",
                                                request=httpx.Request("POST", "https://auth.openai.com/api/accounts/oauth/token")) from None
            raise

    async def _stream_response(self, body, record):
        if record["access_token"] not in self.oauth.secret_values:
            self.oauth.secret_values.append(record["access_token"])
        try:
            async with asyncio.timeout(600):
                return await self._consume_stream(body, record)
        except openai.APIStatusError as exc:
            fields = error_fields(exc)
            message = self.oauth.safe(f"OpenAI HTTP {exc.status_code}; code={fields['code']}; "
                                      f"param={fields['param']}; request_id={fields['request_id']}")
            raise type(exc)(message, response=exc.response, body=exc.body) from None
        except (httpx.TimeoutException, httpcore.TimeoutException, TimeoutError):
            raise openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/responses")) from None
        except (httpx.TransportError, httpcore.NetworkError, httpcore.RemoteProtocolError, ConnectionError) as exc:
            if not is_retryable(exc):
                raise
            raise openai.APIConnectionError(message=f"Responses transport interrupted ({type(exc).__name__}); incomplete output discarded",
                                           request=httpx.Request("POST", "https://api.openai.com/v1/responses")) from None

    async def _consume_stream(self, body, record):
        output_items = {}
        stream = await self._client.responses.create(
            **body, extra_headers={"Authorization": "Bearer " + record["access_token"]})
        http_response = getattr(stream, "response", None)
        headers = getattr(http_response, "headers", {}) or {}
        request_id = headers.get("x-request-id")
        completed = None
        try:
            async with stream:
                async for event in stream:
                    if event.type == "response.output_item.done":
                        output_items[event.output_index] = event.item
                    if event.type == "response.completed":
                        completed = event.response
                        if not completed.output and output_items:
                            completed = completed.model_copy(update={"output": [
                                output_items[index] for index in sorted(output_items)]})
                        return completed
                    if event.type == "response.failed":
                        error = event.response.error
                        details = error.model_dump(exclude_none=True) if hasattr(error, "model_dump") else {}
                        raise self._failure(error.code if error else "response.failed", status=200,
                                            param=getattr(error, "param", None), request_id=request_id,
                                            headers=headers, body={"error": details})
                    if event.type == "response.incomplete":
                        reason = getattr(getattr(event.response, "incomplete_details", None), "reason", "unknown")
                        raise self._failure("response.incomplete", status=200, param=reason,
                                            request_id=request_id, headers=headers)
                    if event.type == "error":
                        raise self._failure(getattr(event, "code", None) or "stream_error", status=200,
                                            param=getattr(event, "param", None), request_id=request_id, headers=headers)
        except (httpx.TransportError, httpcore.NetworkError, httpcore.RemoteProtocolError,
                httpcore.TimeoutException, TimeoutError, ConnectionError):
            if completed is not None:
                return completed  # A cleanup failure must not replay a completed response.
            raise
        except openai.APIError as exc:
            if completed is not None and isinstance(exc, openai.APIConnectionError):
                return completed
            if isinstance(exc, (openai.APIStatusError, openai.APIConnectionError)):
                raise
            # Some SDK versions raise APIError for an SSE error object rather
            # than yielding the error event. Recover its exact code as well.
            fields = error_fields(exc)
            raise self._failure(fields["code"] or "stream_error", status=200, param=fields["param"],
                                request_id=fields["request_id"] or request_id, headers=headers,
                                body=getattr(exc, "body", None)) from None
        raise openai.APIConnectionError(message="Responses stream ended without response.completed",
                                        request=httpx.Request("POST", "https://api.openai.com/v1/responses"))

    async def async_completion(self, conversation, **params: Unpack[TurnCompleter.Params]):
        warn_about_non_empty_params(self, **params)
        body = self._request_body(conversation)
        usage_limit_waits = 0
        while True:
            try:
                async with asyncio.timeout(self.retry_config.stop_after):
                    for retry in range(MAX_INVALID_MESSAGE_RETRIES + 1):
                        response = await self._complete_with_retries(body)
                        if response is None:
                            raise OAuthError("OpenAI returned no completed response")
                        try:
                            return self._to_completion(conversation, response)
                        except InvalidAssistantMessage as exc:
                            response_id = getattr(response, "id", None)
                            logger.warning(
                                "OpenAI completed response has no valid RA assistant message "
                                "(attempt=%s/%s, response_id=%s, reason=%s)",
                                retry + 1, MAX_INVALID_MESSAGE_RETRIES + 1,
                                response_id or "unavailable", str(exc),
                            )
                            if retry >= MAX_INVALID_MESSAGE_RETRIES:
                                raise InvalidAssistantMessage(
                                    f"OpenAI produced no valid RA assistant message after "
                                    f"{MAX_INVALID_MESSAGE_RETRIES + 1} completed responses; "
                                    f"last reason: {exc}"
                                ) from None
            except TimeoutError:
                raise OAuthError("OpenAI request/retry deadline exhausted; no partial output executed") from None
            except (openai.APIStatusError, OpenAIResponseFailure, OAuthHTTPError) as exc:
                if not self.wait_for_account_usage_limit or not is_account_usage_limit(exc):
                    raise
                usage_limit_waits += 1
                delay = max(300.0, retry_after(exc) or 0.0)
                fields = error_fields(exc)
                logger.warning(
                    "OpenAI account usage limit; waiting %.0fs before retry %s "
                    "(code=%s, request_id=%s)",
                    delay, usage_limit_waits, fields["code"], fields["request_id"],
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _complete_with_retries(self, body):
        recovered_auth = False
        pending_refresh = None
        async for attempt in self.retry_config.build():
            with attempt:
                if pending_refresh is not None:
                    record = await self._token(force=True, failed_access_token=pending_refresh)
                    pending_refresh = None
                else:
                    record = await self._token()
                try:
                    return await self._stream_response(body, record)
                except openai.AuthenticationError:
                    if recovered_auth:
                        raise
                    recovered_auth = True
                    pending_refresh = record["access_token"]
                    record = await self._token(force=True, failed_access_token=pending_refresh)
                    pending_refresh = None
                    return await self._stream_response(body, record)
                except openai.BadRequestError as exc:
                    if getattr(exc, "code", None) in {"context_length_exceeded", "input_context_length_exceeded"}:
                        raise self._failure(exc.code) from None
                    raise
        raise OAuthError("OpenAI retries exhausted")

    def _to_completion(self, conversation, response):
        calls, text, refusals, reasoning_items = [], [], [], []
        call_items, call_reasoning = {}, {}
        for item in response.output:
            if item.type == "reasoning":
                if getattr(item, "encrypted_content", None):
                    reasoning_items.append(item.model_dump(exclude_none=True))
            elif item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        text.append(content.text)
                    elif content.type == "refusal":
                        refusals.append(content.refusal)
            elif item.type == "function_call":
                name = item.name.removeprefix(TOOL_NAMESPACE + ".")
                calls.append({"id": item.call_id, "type": "function",
                              "function": {"name": name, "arguments": item.arguments}})
                call_items[item.call_id] = item.model_dump(exclude_none=True)
                call_reasoning[item.call_id] = list(reasoning_items)
            else:
                raise OAuthError(f"Unsupported Responses output: {item.type}")
        if not calls and not text and not refusals:
            raise InvalidAssistantMessage("no text, refusal, or tool calls")
        message = ChatCompletionMessage(role="assistant", content="\n".join(text) or None,
                                        refusal="\n".join(refusals) or None, tool_calls=calls or None)
        if not is_chat_completion_message_param_list(
            [chat_completion_message_to_message_param(message)]
        ):
            raise InvalidAssistantMessage("converted message failed RA's assistant-message format check")
        self._call_items.update(call_items)
        self._reasoning_items.update(call_reasoning)
        return self.Completion(input_conversation=conversation, output_messages=[message], usage=response.usage)


class OpenAIOAuthTurnCompleterConfig(BasicAgentTurnCompleterConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True,
                             json_encoders={NotGiven: lambda _: "NOT_GIVEN"})
    provider: Literal["openai-oauth"] = "openai-oauth"
    model: str = DEFAULT_MODEL
    credentials_dir: str = Field(default_factory=default_credentials_dir)
    proxy: str | None = Field(default_factory=default_proxy)
    context_window: int = Field(default=128_000, gt=0)
    reasoning: Any = NOT_GIVEN
    retry_config: RetryConfig = Field(default_factory=OpenAIOAuthRetryConfig)
    wait_for_account_usage_limit: bool = True

    def build(self):
        return OpenAIOAuthTurnCompleter(
            model=self.model, credentials_dir=self.credentials_dir, proxy=self.proxy,
            context_window=self.context_window, reasoning=self.reasoning,
            tools=[dict(tool.get_oai_tool_call()) for tool in self.basicagent_tools or []],
            retry_config=self.retry_config,
            wait_for_account_usage_limit=self.wait_for_account_usage_limit)
