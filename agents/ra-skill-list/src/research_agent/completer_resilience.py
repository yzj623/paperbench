"""Local resilience for malformed provider tool-call arguments.

This module also absorbs two failure modes specific to providers that police
repeated tool calls (e.g. Alibaba DashScope):

* Layer A (preventive): a tracker observes consecutive identical tool calls
  and, once a threshold is reached, the caller appends a stop-repeating user
  message so the model changes strategy before the provider rejects the
  request.
* Layer B (recovery): when the provider still answers with HTTP 400
  "Repetitive tool calls detected", the caller removes earlier duplicate
  tool-call units from the conversation history and retries with a short
  explanation instead of crashing the whole run.
"""

from __future__ import annotations

import json
from typing import Any

from paperbench.solvers.basicagent.api import CompleterResponse, make_completer_request

_MAX_MALFORMED_TOOL_CALL_RETRIES = 3
_REPAIR_MESSAGE = (
    "The previous response contained a tool call with invalid JSON arguments. "
    "Retry the same action now. Every tool call must include a non-empty valid JSON "
    "object in its arguments; do not emit an empty arguments string."
)


def _error_text(exc: BaseException) -> str:
    """Best-effort extraction of the provider error text from an exception."""

    parts: list[str] = []
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        for key in ("message", "code", "type"):
            value = error.get(key)
            if isinstance(value, str):
                parts.append(value)
    parts.append(str(exc))
    return " ".join(parts)


def is_repetitive_tool_call_error(exc: BaseException) -> bool:
    """Return True for the provider's 400 repetitive-tool-call rejection."""

    if getattr(exc, "status_code", None) != 400:
        return False
    text = _error_text(exc).lower()
    return "repetitive tool call" in text or "repeated across multiple consecutive rounds" in text


class RepetitionTracker:
    """Counts consecutive identical tool calls for the preventive layer."""

    def __init__(self, *, threshold: int = 5):
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.threshold = threshold
        self._signature: str | None = None
        self._count = 0
        self._warned_for_signature: str | None = None

    @staticmethod
    def _signature_of(call: Any) -> str:
        return json.dumps(
            [call.name, dict(call.arguments)], sort_keys=True, ensure_ascii=False, default=str
        )

    def observe(self, call: Any) -> bool:
        """Record one executed call; return True when a warning should be sent.

        The warning is emitted once per continuous run of identical calls, at
        the moment the count first reaches the threshold.
        """

        signature = self._signature_of(call)
        if signature == self._signature:
            self._count += 1
        else:
            self._signature = signature
            self._count = 1
            self._warned_for_signature = None
        if self._count >= self.threshold and self._warned_for_signature != signature:
            self._warned_for_signature = signature
            return True
        return False

    def reset(self) -> None:
        self._signature = None
        self._count = 0
        self._warned_for_signature = None


def build_repetition_warning(call: Any, *, count: int) -> str:
    """User message that asks the model to stop repeating a tool call."""

    try:
        arguments = json.dumps(dict(call.arguments), ensure_ascii=False)
    except (TypeError, ValueError):
        arguments = str(getattr(call, "arguments", {}))
    return (
        f"You have now called the same tool ({getattr(call, 'name', 'unknown')!r}) with identical "
        f"arguments {count} times in a row: {arguments}\n"
        "Stop repeating this exact call. A repeated call will not produce new information. "
        "Either change the command/arguments to obtain the missing information, or accept the "
        "result you already have and continue with the next step of the current subtask. "
        "Do not issue this identical call again."
    )


async def make_resilient_completer_request(
    *, completer: Any, messages: list[dict[str, Any]]
) -> CompleterResponse:
    """Retry provider responses whose tool-call arguments are not valid JSON."""

    for attempt in range(_MAX_MALFORMED_TOOL_CALL_RETRIES + 1):
        try:
            return await make_completer_request(completer=completer, messages=messages)
        except json.JSONDecodeError:
            if attempt >= _MAX_MALFORMED_TOOL_CALL_RETRIES:
                raise
            messages.append({"role": "user", "content": _REPAIR_MESSAGE})
    raise AssertionError("unreachable")
