"""Protected plan-at-the-top conversation memory and safe trimming."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

Message = dict[str, Any]


def _is_tool_call(message: Message) -> bool:
    return message.get("role") == "assistant" and bool(message.get("tool_calls"))


def _split_units(messages: list[Message]) -> list[list[Message]]:
    """Split history into units without leaving an assistant/tool orphan."""

    units: list[list[Message]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if _is_tool_call(message):
            call_ids = {
                call.get("id")
                for call in message.get("tool_calls", [])
                if isinstance(call, dict) and call.get("id")
            }
            unit = [message]
            index += 1
            while index < len(messages):
                candidate = messages[index]
                if candidate.get("role") != "tool" or candidate.get("tool_call_id") not in call_ids:
                    break
                unit.append(candidate)
                index += 1
            units.append(unit)
            continue
        # Orphan tool messages are not valid API history and are discarded.
        if message.get("role") == "tool":
            index += 1
            continue
        units.append([message])
        index += 1
    return units


def trim_message_units(
    history: Iterable[Message], *, max_messages: int, remove_units: int | None = None
) -> list[Message]:
    """Remove the earliest complete units until the history fits.

    The first two protected messages are handled by :class:`MessageWindow` and
    are never passed to this function.
    """

    if max_messages < 0:
        raise ValueError("max_messages must be non-negative")
    units = _split_units(list(history))
    if remove_units is None:
        while units and sum(len(unit) for unit in units) > max_messages:
            units.pop(0)
    else:
        if remove_units < 0:
            raise ValueError("remove_units must be non-negative")
        del units[:remove_units]
    return [message for unit in units for message in unit]


class MessageWindow:
    """Conversation with protected system, live-plan, and optional skill slots."""

    def __init__(self, system_message: Message, plan_message: Message):
        if system_message.get("role") != "system":
            raise ValueError("system_message must have role=system")
        if plan_message.get("role") != "user":
            raise ValueError("plan_message must have role=user")
        self._system = dict(system_message)
        self._plan = dict(plan_message)
        self._skill: Message | None = None
        self._history: list[Message] = []
        self.audit_log = None

    @property
    def messages(self) -> list[Message]:
        protected = [dict(self._system), dict(self._plan)]
        if self._skill is not None:
            protected.append(dict(self._skill))
        return [*protected, *[dict(item) for item in self._history]]

    @property
    def history(self) -> list[Message]:
        return [dict(item) for item in self._history]

    def update_plan(self, content: str | dict[str, Any]) -> None:
        self._plan = {"role": "user", "content": content}
    def update_skill(self, content: str | None) -> None:
        """Replace the protected skill slot without touching execution history."""

        self._skill = None if content is None else {"role": "user", "content": content}


    def append(self, message: Message) -> None:
        if message.get("role") in {"system"}:
            raise ValueError("protected system message cannot be appended")
        self._history.append(dict(message))

    def extend(self, messages: Iterable[Message]) -> None:
        for message in messages:
            self.append(message)

    def trim_for_context(self, max_messages: int) -> None:
        self._history = trim_message_units(self._history, max_messages=max_messages)

    def trim_oldest_unit(self) -> bool:
        before_messages = self.messages
        before = len(self._history)
        self._history = trim_message_units(self._history, max_messages=10**18, remove_units=1)
        if len(self._history) < before:
            from .audit import record
            record(self.audit_log, "context_unit_removed",
                   removed_messages=before_messages[2:2 + before-len(self._history)],
                   retained_messages=self.messages)
        return len(self._history) < before

    def drop_duplicate_tool_call_units(self) -> int:
        """Remove earlier copies of identical tool-call units, keeping the last.

        A "unit" is one assistant message carrying tool_calls together with its
        matching tool responses (see :func:`_split_units`). Only units whose
        tool calls are byte-for-byte identical (same name and same arguments)
        are collapsed; the last occurrence is retained. This directly targets
        providers that reject a conversation history containing many identical
        consecutive tool calls, without disturbing assistant/tool pairing.
        """

        import json

        def _signature(unit: list[Message]) -> str | None:
            if not unit or not _is_tool_call(unit[0]):
                return None
            calls = [
                (call.get("name"), json.dumps(call.get("arguments"), sort_keys=True))
                for call in unit[0].get("tool_calls", [])
                if isinstance(call, dict)
            ]
            return json.dumps(calls, sort_keys=True) if calls else None

        units = _split_units(self._history)
        last_index: dict[str, int] = {}
        for index, unit in enumerate(units):
            signature = _signature(unit)
            if signature is not None:
                last_index[signature] = index
        kept: list[list[Message]] = []
        removed_units = 0
        for index, unit in enumerate(units):
            signature = _signature(unit)
            if signature is not None and last_index[signature] != index:
                removed_units += 1
                continue
            kept.append(unit)
        if removed_units:
            self._history = [message for unit in kept for message in unit]
        return removed_units
