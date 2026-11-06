"""Bounded, credential-safe retries for public ChatGPT-plan Responses requests.

Policy references: OpenAI SIWC errors-and-recovery and API rate-limits guides.
This module is intentionally provider-local: Qwen/DeepSeek retries are untouched.
"""
from __future__ import annotations

from email.utils import parsedate_to_datetime
import logging
import math
import random
import time
from datetime import timezone

import httpcore
import httpx
import openai
import tenacity
from paperbench.solvers.basicagent.completer import TimeTrackingRetryConfig
from pydantic import Field

from .openai_oauth import OAuthError

LOGGER = logging.getLogger(__name__)
TRANSIENT_CODES = {
    "server_error", "internal_error", "internal_server_error", "timeout", "request_timeout",
    "rate_limit_exceeded", "rate_limit_error", "slow_down", "server_is_overloaded",
    "service_unavailable", "service_unavailable_error", "temporarily_unavailable",
    "subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable",
}
PERMANENT_CODES = {
    "insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached", "billing_not_active",
    "organization_spend_limit_exceeded", "project_spend_limit_exceeded", "organization_usage_limit_exceeded",
    "quota_exceeded", "subscription_sharing_usage_limit_exceeded", "subscription_sharing_user_not_eligible",
    "subscription_sharing_unsupported_capability", "subscription_sharing_route_not_supported",
    "subscription_sharing_invalid_user", "chatpass_v2_scope_not_authorized",
    "chatpass_v2_invalid_authorization_context", "misalignment_policy_violation", "content_filter",
    "invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired",
    "refresh_token_invalidated", "refresh_token_reused", "invalid_client",
}


def error_fields(exc):
    body = getattr(exc, "body", None)
    body = body if isinstance(body, dict) else {}
    error = body.get("error", body)
    error = error if isinstance(error, dict) else {}
    code = getattr(exc, "code", None) or error.get("code") or error.get("type")
    code = code if isinstance(code, str) else None
    response = getattr(exc, "response", None)
    headers = getattr(exc, "headers", None) or getattr(response, "headers", {}) or {}
    return {"status": getattr(exc, "status_code", None), "code": code,
            "param": getattr(exc, "param", None) or error.get("param"), "headers": headers,
            "request_id": getattr(exc, "request_id", None) or headers.get("x-request-id")}


class OpenAIResponseFailure(OAuthError):
    """A stream error retains metadata but never includes prompt/token content in str()."""
    def __init__(self, code, *, status=None, param=None, request_id=None, headers=None, body=None):
        super().__init__(f"OpenAI Responses failed: code={code}; status={status}; "
                         f"param={param}; request_id={request_id}")
        self.code, self.status_code, self.param = code, status, param
        self.request_id, self.headers = request_id, headers or {}
        self.body = body or {"error": {"code": code, "param": param}}


def retry_after(exc, *, now=None):
    headers = error_fields(exc)["headers"]
    for name, divisor in [("retry-after-ms", 1000), ("retry-after", 1)]:
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            seconds = float(raw) / divisor
        except (ValueError, TypeError):
            if name != "retry-after":
                continue
            try:
                date = parsedate_to_datetime(raw)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                seconds = max(0, date.timestamp() - (time.time() if now is None else now))
            except (ValueError, TypeError, OverflowError):
                continue
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return None


def is_retryable(exc):
    fields = error_fields(exc)
    if fields["code"] in PERMANENT_CODES or fields["headers"].get("x-should-retry") == "false":
        return False
    if isinstance(exc, (httpx.LocalProtocolError, httpx.UnsupportedProtocol, httpcore.LocalProtocolError)):
        return False  # Invalid local request construction isn't a network interruption.
    if isinstance(exc, (httpx.TransportError, httpcore.NetworkError, httpcore.RemoteProtocolError,
                        httpcore.TimeoutException, openai.APIConnectionError, TimeoutError, ConnectionError)):
        return True
    status = fields["status"]
    if status in {400, 401, 403, 404, 422}:
        return False
    if fields["code"] in TRANSIENT_CODES:
        return True
    if status is not None:
        return status in {408, 409, 429} or status >= 500
    # SDK stream readers can raise APIError without an HTTP status, after HTTP 200.
    return fields["code"] in TRANSIENT_CODES


class OpenAIOAuthRetryConfig(TimeTrackingRetryConfig):
    wait_min: float = Field(default=1, ge=0)
    wait_max: float = Field(default=300, ge=0)
    stop_after: float = Field(default=7200, gt=0)
    max_attempts: int = Field(default=12, ge=1)

    def build(self):
        exponential = tenacity.wait_random_exponential(min=self.wait_min, max=self.wait_max)

        def eligible(exc):
            if not is_retryable(exc):
                return False
            delay = retry_after(exc)
            if delay is not None and delay > self.wait_max:
                fields = error_fields(exc)
                LOGGER.warning("OpenAI retry deferred: server delay=%s exceeds cap=%s; status=%s code=%s request_id=%s",
                               delay, self.wait_max, fields["status"], fields["code"], fields["request_id"])
                return False  # Never retry sooner than a server-specified minimum.
            return True

        def wait(state):
            base = exponential(state)
            delay = retry_after(state.outcome.exception())
            if delay is None:
                return base
            return min(self.wait_max, max(base, delay) + random.uniform(0, min(1, self.wait_max - delay)))

        def record(state):
            delay = state.next_action.sleep if state.next_action else 0
            self.time_spent_retrying += delay
            exc = state.outcome.exception()
            fields = error_fields(exc)
            LOGGER.warning("OpenAI retry %s/%s in %.2fs: error=%s status=%s code=%s request_id=%s",
                           state.attempt_number + 1, self.max_attempts, delay, type(exc).__name__,
                           fields["status"], fields["code"], fields["request_id"])

        return tenacity.AsyncRetrying(wait=wait, stop=tenacity.stop_after_attempt(self.max_attempts) |
                                     tenacity.stop_before_delay(self.stop_after),
                                     retry=tenacity.retry_if_exception(eligible), before_sleep=record, reraise=True)


def oauth_retry_config(config=None):
    if isinstance(config, OpenAIOAuthRetryConfig):
        return config
    if config is None:
        return OpenAIOAuthRetryConfig()
    return OpenAIOAuthRetryConfig(wait_min=config.wait_min, wait_max=config.wait_max, stop_after=config.stop_after,
                                 max_attempts=getattr(config, "max_attempts", 12))
