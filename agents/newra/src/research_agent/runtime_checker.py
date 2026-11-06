"""Concrete stateless Check Agent implementation."""

from __future__ import annotations

from .audit import audited_request

import json
from pathlib import Path
from typing import Any, Sequence

import tiktoken
from openai import LengthFinishReasonError
from openai.types.responses import FunctionToolParam

from paperbench.solvers.basicagent.api import make_completer_request
from paperbench.solvers.basicagent.tools.base import Tool

from .api_settings import CompatibleAPISettings
from .checker import (
    CheckAgentContext,
    CheckerFactory,
    CheckerResponseError,
    StatelessChecker,
    parse_check_report,
)
from .context import MessageWindow
from .prompts import CHECKER_PROMPT, render_workspace_prompt
from .runtime_tools import ReadOnlyAgentTool
from .full_review import FullReviewReport
from .prompts import FULL_REVIEW_PROMPT

_CONTEXT_TRIM_TRIGGER_RATIO = 0.60
_CONTEXT_TRIM_TARGET_RATIO = 0.45


def _log_value(value: Any, *, limit: int = 8_000) -> str:
    """Keep tool logs useful without duplicating huge file contents."""

    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text


def _estimate_request_tokens(*, completer: Any, messages: Sequence[dict[str, Any]]) -> int:
    """Estimate the compatible-chat request size with the configured tokenizer."""

    encoding_name = str(getattr(completer, "encoding_name", "o200k_base"))
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError:
        encoding = tiktoken.get_encoding("o200k_base")
    payload: dict[str, Any] = {"messages": list(messages)}
    configured_tools = getattr(completer, "tools", None)
    if isinstance(configured_tools, list):
        payload["tools"] = configured_tools
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    return len(encoding.encode_ordinary(serialized))


def _trim_checker_context(
    *,
    window: MessageWindow,
    completer: Any,
    context_window: int,
    force: bool = False,
) -> tuple[int, int, int]:
    """Trim the checker window proactively (60% trigger / 45% target).

    Mirrors the main agent's budgeted trimming so the stateless Check Agent
    avoids repeated context-overflow rejections while exploring evidence.
    """

    before = _estimate_request_tokens(completer=completer, messages=window.messages)
    trigger = max(1, int(context_window * _CONTEXT_TRIM_TRIGGER_RATIO))
    target = max(1, int(context_window * _CONTEXT_TRIM_TARGET_RATIO))
    if not force and before < trigger:
        return before, before, 0

    removed = 0
    current = before
    if force and current <= target:
        # The provider reported an overflow even though the estimate was low,
        # so drop at least one complete unit before re-checking.
        if window.trim_oldest_unit():
            removed += 1
            current = _estimate_request_tokens(completer=completer, messages=window.messages)
    while current > target:
        if not window.trim_oldest_unit():
            break
        removed += 1
        current = _estimate_request_tokens(completer=completer, messages=window.messages)
    return before, current, removed


class RuntimeChecker(StatelessChecker):
    """A fresh tool-using checker created for one subtask submission."""

    def __init__(
        self,
        *,
        completer_config: CompatibleAPISettings,
        context: CheckAgentContext,
        max_steps: int = 300,
    ):
        if context.read_only_dispatcher is None:
            raise ValueError("Check Agent requires a read-only dispatcher")
        self.api_settings = completer_config
        self.context = context
        self.max_steps = max_steps

    def _tools(self) -> list[Tool]:
        dispatcher = self.context.read_only_dispatcher
        read_only: list[Tool] = [
            ReadOnlyAgentTool(
                tool_name=name,
                description=str(schema["function"].get("description", name)),
                dispatcher=dispatcher,
            )
            for name, schema in dispatcher.schemas.items()
        ]
        return [*read_only, SubmitFullReviewTool() if self.context.full_review else SubmitReviewTool()]

    def _build_completer(
        self,
        *,
        tools: list[Tool],
    ) -> Any:
        """Build a fresh role-local checker config for one protocol phase."""

        return self.api_settings.new_config(
            basicagent_tools=tools,
            response_format=None,
        ).build()

    async def review(self, context: CheckAgentContext) -> str:
        tools = self._tools()
        # Exploration must be allowed to use the read-only tools.  Enabling
        # JSON mode here makes some compatible models skip exploration and
        # emit a plausible but evidence-free report immediately.
        completer = self._build_completer(tools=tools)
        dispatcher_workspace = getattr(self.context.read_only_dispatcher, "workspace", None)
        workspace_root = getattr(dispatcher_workspace, "workspace_root", None)
        workspace = Path(workspace_root).as_posix() if workspace_root is not None else "/home"
        if context.full_review:
            briefing = (
                "Review the full current submission against all plan subtasks and the public paper. "
                "Use read-only tools to inspect actual files. Prior rounds and primary-agent responses "
                "are historical evidence, not current truth.\n"
                f"Plan and statuses: {json.dumps(list(context.plan), ensure_ascii=False)}\n"
                f"Previous rounds: {json.dumps(list(context.previous_rounds), ensure_ascii=False)}"
            )
        else:
            briefing = (
                    "Review this frozen subtask. Your only acceptance question is whether this "
                    "subtask was reproduced precisely; the briefing at /check_instruction.txt, "
                    "the plan, and the statuses are context. Inspect the actual public paper "
                    "and workspace with the read-only tools.\n\n"
                    f"Subtask: {context.task.model_dump()}\n"
                    "Current frozen subtask (including runtime-owned status):\n"
                    f"{json.dumps(list(context.plan), ensure_ascii=False, indent=2, default=str)}\n"
                    f"Primary-agent evidence pointer: {context.reproduction_description}"
            )
        window = MessageWindow(
            {"role": "system", "content": render_workspace_prompt(
                FULL_REVIEW_PROMPT if context.full_review else CHECKER_PROMPT, workspace)},
            {"role": "user", "content": briefing},
        )
        window.audit_log = self.context.tool_log
        by_name = {tool.name(): tool for tool in tools}
        invalid_reviews = 0
        context_window = getattr(completer, "context_window", None) or getattr(
            completer, "n_ctx", None
        )

        def _record_trim(
            before_tokens: int | None,
            after_tokens: int | None,
            removed: int,
            *,
            proactive: bool,
            request_attempt: int | None = None,
        ) -> None:
            if self.context.tool_log is None or removed <= 0:
                return
            record: dict[str, Any] = {
                "estimated_tokens_before": before_tokens,
                "estimated_tokens_after": after_tokens,
                "removed_units": removed,
                "proactive": proactive,
            }
            if proactive:
                record["trigger_ratio"] = _CONTEXT_TRIM_TRIGGER_RATIO
                record["target_ratio"] = _CONTEXT_TRIM_TARGET_RATIO
            if request_attempt is not None:
                record["request_attempt"] = request_attempt
            self.context.tool_log.append("checker_context_trim", **record)

        for _ in range(self.max_steps):
            if context_window:
                before_tokens, after_tokens, removed_units = _trim_checker_context(
                    window=window,
                    completer=completer,
                    context_window=int(context_window),
                )
                _record_trim(
                    before_tokens,
                    after_tokens,
                    removed_units,
                    proactive=True,
                )
            try:
                result = await audited_request(make_completer_request, audit_log=self.context.tool_log, role="checker",
                    completer=completer, messages=window.messages
                )
            except (LengthFinishReasonError, IndexError):
                if context_window:
                    before_tokens, after_tokens, removed_units = _trim_checker_context(
                        window=window,
                        completer=completer,
                        context_window=int(context_window),
                        force=True,
                    )
                else:
                    removed_units = int(window.trim_oldest_unit())
                    before_tokens = after_tokens = None
                if not removed_units:
                    raise
                _record_trim(
                    before_tokens,
                    after_tokens,
                    removed_units,
                    proactive=False,
                )
                continue
            window.extend(result.response_messages)
            if not result.tool_calls:
                window.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue the read-only review. When your judgment is ready, you must "
                            f"call {'submit_full_review' if context.full_review else 'submit_review'}; "
                            "ordinary assistant text cannot finish the review."
                        ),
                    }
                )
                continue
            for call in result.tool_calls:
                if self.context.tool_log is not None:
                    self.context.tool_log.append(
                        "checker_tool_call",
                        tool=call.name,
                        arguments=dict(call.arguments),
                    )
                if call.name == ("submit_full_review" if context.full_review else "submit_review"):
                    try:
                        report = (FullReviewReport.model_validate(call.arguments) if context.full_review
                                  else parse_check_report(call.arguments))
                    except Exception as exc:
                        invalid_reviews += 1
                        output = json.dumps(
                            {"ok": False, "error": "INVALID_REVIEW", "details": str(exc)},
                            ensure_ascii=False,
                        )
                        if self.context.tool_log is not None:
                            self.context.tool_log.append(
                                "checker_review_invalid", details=str(exc)
                            )
                        window.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.call_id,
                                "content": output,
                            }
                        )
                        if invalid_reviews >= 3:
                            raise RuntimeError(
                                "Check Agent submitted three invalid reviews"
                            ) from exc
                        continue
                    if self.context.tool_log is not None:
                        self.context.tool_log.append(
                            "checker_review_submitted",
                            accepted=(report.accepted if not context.full_review else not report.issues),
                            summary=(report.summary if not context.full_review else "full submission review"),
                            issues=[issue.model_dump() for issue in report.issues],
                        )
                    return report.model_dump()
                tool = by_name.get(call.name)
                if tool is None:
                    output = f"ERROR: unsupported read-only tool {call.name!r}"
                    if self.context.tool_log is not None:
                        self.context.tool_log.append(
                            "checker_tool_error", tool=call.name, error=output
                        )
                else:
                    try:
                        output = await tool.execute(None, **call.arguments)
                    except Exception as exc:
                        output = f"ERROR: {type(exc).__name__}: {exc}"
                        if self.context.tool_log is not None:
                            self.context.tool_log.append(
                                "checker_tool_error",
                                tool=call.name,
                                error=output,
                            )
                    else:
                        if self.context.tool_log is not None:
                            self.context.tool_log.append(
                                "checker_tool_result",
                                tool=call.name,
                                result=_log_value(output),
                            )
                window.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": output,
                    }
                )
        raise RuntimeError("Check Agent exceeded its read-only tool-call limit")


class SubmitReviewTool(Tool):
    """Terminal Check Agent protocol tool; execution is intercepted by RuntimeChecker."""

    def name(self) -> str:
        return "submit_review"

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description=(
                "Submit the final subtask fidelity judgment after inspecting the actual evidence. "
                "This ends the Check Agent review when the arguments are schema-valid."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "accepted": {"type": "boolean"},
                    "summary": {"type": "string", "minLength": 1},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string", "minLength": 1},
                                "evidence": {"type": "string", "minLength": 1},
                            },
                            "required": ["description", "evidence"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["accepted", "summary", "issues"],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, computer: Any, **kwargs: Any) -> str:
        del computer, kwargs
        raise RuntimeError("submit_review must be intercepted by RuntimeChecker")


class SubmitFullReviewTool(Tool):
    """Terminal protocol for one whole-submission review."""

    def name(self) -> str:
        return "submit_full_review"

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function", name=self.name(), strict=False,
            description=("Submit all concrete current-submission issues once. Empty issues means no "
                         "reportable problems. Keep an old issue_id for an unresolved old issue; "
                         "assign a new unique id to a newly discovered issue."),
            parameters={"type": "object", "properties": {"issues": {"type": "array", "items": {
                "type": "object", "properties": {
                    key: {"type": "string", "minLength": 1}
                    for key in ("issue_id", "description", "paper_evidence", "code_evidence")},
                "required": ["issue_id", "description", "paper_evidence", "code_evidence"],
                "additionalProperties": False}}}, "required": ["issues"],
                "additionalProperties": False},
        )

    async def execute(self, computer: Any, **kwargs: Any) -> str:
        raise RuntimeError("submit_full_review is intercepted by RuntimeChecker")


class RuntimeCheckerFactory(CheckerFactory):
    """Builds a new RuntimeChecker while reusing the configured model."""

    def __init__(self, *, max_steps: int = 300):
        self.max_steps = max_steps

    async def build(self, *, completer_config: Any, context: CheckAgentContext) -> RuntimeChecker:
        return RuntimeChecker(
            completer_config=completer_config,
            context=context,
            max_steps=self.max_steps,
        )
