"""Planning-to-execution session with a fresh context after planning."""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol, Sequence

from .context import MessageWindow
from .plan import Plan, PlanValidationError, format_invalid_plan_error, parse_plan
from .prompts import EXECUTION_PROMPT, PLANNING_PROMPT, render_plan_message
from .storage import PlanStore


class ResearchCompleter(Protocol):
    async def complete(self, messages: Sequence[dict[str, Any]], *, tools: Sequence[Any]) -> Any: ...


class ResearchAgentSession:
    """Coordinates planning and the clean execution context.

    ``complete`` is deliberately injected so local tests can use a fake model;
    the production adapter can pass the existing Alibaba-compatible completer.
    """

    def __init__(
        self,
        *,
        store: PlanStore,
        completer: ResearchCompleter,
        planning_prompt: str = PLANNING_PROMPT,
        execution_prompt: str = EXECUTION_PROMPT,
    ):
        self.store = store
        self.completer = completer
        self.planning_prompt = planning_prompt
        self.execution_prompt = execution_prompt
        self.window: MessageWindow | None = None

    async def create_plan(
        self, *, public_material_message: str, max_retries: int | None = None
    ) -> Plan:
        if max_retries is not None and max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.planning_prompt},
            {"role": "user", "content": public_material_message},
        ]
        last_error: PlanValidationError | None = None
        attempts = 0
        # ``max_retries`` counts repairs after the initial planning attempt.
        while max_retries is None or attempts <= max_retries:
            attempts += 1
            response = await self.completer.complete(messages, tools=[])
            raw = response if isinstance(response, (str, Mapping)) else getattr(response, "content", None)
            try:
                plan = parse_plan(raw)
            except PlanValidationError as exc:
                last_error = exc
                messages.append({"role": "assistant", "content": str(raw)})
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            format_invalid_plan_error(exc), ensure_ascii=False
                        ),
                    }
                )
                continue
            self.store.initialize(plan)
            # The planning conversation is intentionally discarded.  Execution
            # starts with only the fixed system prompt and the live plan.
            self.window = MessageWindow(
                {"role": "system", "content": self.execution_prompt},
                render_plan_message(self.store.render()),
            )
            self.store.events.append("planning_context_discarded")
            return plan
        assert last_error is not None
        raise last_error

    def sync_plan_message(self) -> None:
        if self.window is None:
            raise RuntimeError("execution context has not been created")
        self.window.update_plan(render_plan_message(self.store.render())["content"])

    def execution_messages(self) -> list[dict[str, Any]]:
        if self.window is None:
            raise RuntimeError("execution context has not been created")
        self.sync_plan_message()
        return self.window.messages
