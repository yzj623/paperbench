"""Concrete stateless Check Agent implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import tiktoken
from openai import LengthFinishReasonError
from openai.types.responses import FunctionToolParam

from paperbench.solvers.basicagent.api import make_completer_request
from paperbench.solvers.basicagent.tools.base import Tool

from .audit import audited_request
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

_CONTEXT_TRIM_TRIGGER_RATIO = 0.60
_CONTEXT_TRIM_TARGET_RATIO = 0.45


class ErrorRecordTool(Tool):
    """Stage a complete replacement for one subtask's active issue record."""

    original_record: list[dict[str, Any]]
    log: Any | None = None
    candidate: list[dict[str, Any]] | None = None
    success_count: int = 0

    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return "update_error_record"

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description=(
                "Update the active error record for this subtask after inspecting the current code. "
                "Provide exactly one keep/delete action with a non-empty reason for every issue in "
                "the original record, plus all newly discovered blocking issues. Keep issue names "
                "concise and requirement-level. Each new or kept issue description must contain: "
                "Requirement, Violation, Evidence, Acceptance, and Scope. For global requirements, "
                "list the affected tasks or interfaces in Scope. Delete an issue only when its "
                "Acceptance condition is met by current code evidence; a disappeared symptom, added "
                "dependency or declaration, wrapper, or local-only change is insufficient unless "
                "the requirement itself concerns that artifact. Do not narrow or replace an issue "
                "through new_name or new_description; for a genuinely new scope, keep the original "
                "issue and add a separate issue. Each invocation is based on the original record "
                "from the start of this checker run. A later successful invocation replaces the "
                "previous candidate; a failed invocation does not. Kept issues retain their order, "
                "new issues are appended in input order, and the result is renumbered from 1. The "
                "candidate is saved only after submit_review succeeds. This tool does not end the "
                "review; call submit_review afterward."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "existing_issue_actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "issue_id": {"type": "integer", "minimum": 1},
                                "action": {"type": "string", "enum": ["keep", "delete"]},
                                "reason": {"type": "string", "minLength": 1},
                                "new_name": {"type": "string", "minLength": 1},
                                "new_description": {"type": "string", "minLength": 1},
                            },
                            "required": ["issue_id", "action", "reason"],
                            "additionalProperties": False,
                        },
                    },
                    "new_issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "description": {"type": "string", "minLength": 1},
                            },
                            "required": ["name", "description"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["existing_issue_actions", "new_issues"],
                "additionalProperties": False,
            },
            strict=False,
        )

    @staticmethod
    def _required_text(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        return value.strip()

    async def execute(self, computer: Any, **kwargs: Any) -> str:
        del computer
        try:
            actions = kwargs.get("existing_issue_actions")
            new_issues = kwargs.get("new_issues")
            if not isinstance(actions, list) or not isinstance(new_issues, list):
                raise ValueError("existing_issue_actions and new_issues must be arrays")
            expected = {item["issue_id"]: item for item in self.original_record}
            seen: set[int] = set()
            kept: list[dict[str, Any]] = []
            audit_actions: list[dict[str, Any]] = []
            for action in actions:
                if not isinstance(action, dict):
                    raise ValueError("each existing issue action must be an object")
                issue_id = action.get("issue_id")
                if not isinstance(issue_id, int) or isinstance(issue_id, bool) or issue_id not in expected:
                    raise ValueError(f"unknown issue_id: {issue_id!r}")
                if issue_id in seen:
                    raise ValueError(f"issue_id {issue_id} appears more than once")
                seen.add(issue_id)
                operation = action.get("action")
                if operation not in {"keep", "delete"}:
                    raise ValueError(f"issue_id {issue_id} action must be keep or delete")
                reason = self._required_text(action.get("reason"), "reason")
                if operation == "keep":
                    prior = expected[issue_id]
                    name = self._required_text(action.get("new_name", prior["name"]), "new_name")
                    description = self._required_text(
                        action.get("new_description", prior["description"]), "new_description"
                    )
                    kept.append({"name": name, "description": description})
                elif "new_name" in action or "new_description" in action:
                    raise ValueError("delete actions may not include new_name or new_description")
                audit_actions.append(
                    {"issue_id": issue_id, "action": operation, "reason": reason}
                )
            if seen != set(expected):
                missing = sorted(set(expected) - seen)
                raise ValueError(f"every original issue requires one action; missing IDs: {missing}")

            additions: list[dict[str, str]] = []
            for item in new_issues:
                if not isinstance(item, dict):
                    raise ValueError("each new issue must be an object")
                additions.append(
                    {
                        "name": self._required_text(item.get("name"), "new issue name"),
                        "description": self._required_text(
                            item.get("description"), "new issue description"
                        ),
                    }
                )
            candidate = [
                {"issue_id": index, **item}
                for index, item in enumerate([*kept, *additions], start=1)
            ]
            superseded = self.success_count > 0
            self.candidate = candidate
            self.success_count += 1
            if self.log is not None:
                self.log.append(
                    "checker_error_record_candidate_updated",
                    original=self.original_record,
                    actions=audit_actions,
                    new_issues=additions,
                    candidate=candidate,
                    superseded_previous_candidate=superseded,
                )
            return json.dumps(
                {
                    "ok": True,
                    "superseded_previous_candidate": superseded,
                    "candidate_error_record": candidate,
                },
                ensure_ascii=False,
            )
        except (KeyError, TypeError, ValueError) as exc:
            if self.log is not None:
                self.log.append(
                    "checker_error_record_update_invalid",
                    original=self.original_record,
                    arguments=kwargs,
                    error=str(exc),
                )
            return json.dumps(
                {"ok": False, "error": "INVALID_ERROR_RECORD_UPDATE", "details": str(exc)},
                ensure_ascii=False,
            )


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
        return [
            *read_only,
            ErrorRecordTool(
                original_record=[dict(item) for item in self.context.error_record],
                log=self.context.tool_log,
            ),
            SubmitReviewTool(),
        ]

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
        window = MessageWindow(
            {"role": "system", "content": render_workspace_prompt(CHECKER_PROMPT, workspace)},
            {
                "role": "user",
                "content": (
                    "Review this frozen subtask. Your only acceptance question is whether this "
                    "subtask was reproduced precisely; the briefing at /check_instruction.txt, "
                    "the plan, and the statuses are context. Inspect the actual public paper "
                    "and workspace with the read-only tools.\n\n"
                    f"Subtask: {context.task.model_dump()}\n"
                    "Current frozen subtask (including runtime-owned status):\n"
                    f"{json.dumps(list(context.plan), ensure_ascii=False, indent=2, default=str)}\n"
                    f"Primary-agent evidence pointer: {context.reproduction_description}"
                ),
            },
        )
        window.audit_log = self.context.tool_log
        if context.error_record:
            window.update_error_record(
                "Persistent active error record for this subtask:\n"
                f"{json.dumps(list(context.error_record), ensure_ascii=False, indent=2)}\n\n"
                "This is historical guidance, not current evidence. The workspace may have changed. "
                "Re-inspect the current implementation before keeping or deleting each issue, and "
                "independently check for new blocking problems."
            )
        by_name = {tool.name(): tool for tool in tools}
        error_record_tool = by_name["update_error_record"]

        def _log_abandoned(reason: str) -> None:
            if self.context.tool_log is not None and error_record_tool.candidate is not None:
                self.context.tool_log.append(
                    "checker_error_record_update_abandoned",
                    task_id=context.task.id,
                    candidate=error_record_tool.candidate,
                    reason=reason,
                )
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
                result = await audited_request(
                    make_completer_request,
                    audit_log=self.context.tool_log,
                    role="checker",
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
                    _log_abandoned("context overflow could not be resolved by trimming")
                    raise
                _record_trim(
                    before_tokens,
                    after_tokens,
                    removed_units,
                    proactive=False,
                )
                continue
            except BaseException as exc:
                _log_abandoned(f"checker execution ended before submit_review: {type(exc).__name__}")
                raise
            window.extend(result.response_messages)
            if not result.tool_calls:
                if error_record_tool.success_count == 0:
                    reminder = (
                        "Continue the read-only review. This review has not yet successfully called "
                        "update_error_record. When ready, call update_error_record first and then "
                        "submit_review. Ordinary assistant text cannot finish the review."
                    )
                else:
                    reminder = (
                        "A candidate active error record has been produced, but this review has not "
                        "successfully called submit_review. You may replace the candidate by calling "
                        "update_error_record again, or finish with a consistent submit_review call. "
                        "Ordinary assistant text cannot finish the review."
                    )
                window.append(
                    {
                        "role": "user",
                        "content": reminder,
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
                if call.name == "submit_review":
                    try:
                        report = parse_check_report(call.arguments)
                    except CheckerResponseError as exc:
                        invalid_reviews += 1
                        output = json.dumps(
                            {"ok": False, "error": "INVALID_REVIEW", "details": exc.details},
                            ensure_ascii=False,
                        )
                        if self.context.tool_log is not None:
                            self.context.tool_log.append(
                                "checker_review_invalid", details=exc.details
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
                    candidate = error_record_tool.candidate
                    if error_record_tool.success_count == 0 or candidate is None:
                        if self.context.tool_log is not None:
                            self.context.tool_log.append(
                                "checker_review_missing_error_record_update",
                                task_id=context.task.id,
                            )
                        output = json.dumps(
                            {
                                "ok": False,
                                "error": "ERROR_RECORD_UPDATE_REQUIRED",
                                "details": "Call update_error_record successfully before submit_review.",
                            },
                            ensure_ascii=False,
                        )
                        window.append(
                            {"role": "tool", "tool_call_id": call.call_id, "content": output}
                        )
                        continue
                    if report.accepted != (len(candidate) == 0):
                        output = json.dumps(
                            {
                                "ok": False,
                                "error": "REVIEW_RECORD_CONFLICT",
                                "details": (
                                    "accepted=true requires an empty candidate error record; "
                                    "accepted=false requires a non-empty candidate error record."
                                ),
                            },
                            ensure_ascii=False,
                        )
                        if self.context.tool_log is not None:
                            self.context.tool_log.append(
                                "checker_review_record_conflict",
                                accepted=report.accepted,
                                candidate=candidate,
                            )
                        window.append(
                            {"role": "tool", "tool_call_id": call.call_id, "content": output}
                        )
                        continue
                    if self.context.error_record_store is not None:
                        self.context.error_record_store.commit_error_record(
                            context.task.id,
                            candidate,
                            reason=report.summary,
                        )
                    if self.context.tool_log is not None:
                        self.context.tool_log.append(
                            "checker_review_submitted",
                            accepted=report.accepted,
                            summary=report.summary,
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
        _log_abandoned("checker exceeded its read-only tool-call limit")
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
                "Submit the final fidelity judgment after inspecting the current implementation. "
                "You must first successfully call update_error_record. accepted=true requires its "
                "latest candidate record to be empty and issues=[]; accepted=false requires a "
                "non-empty candidate record and at least one issue. On rejection, issues remain "
                "the feedback delivered to the primary agent and should faithfully describe the "
                "blocking problems in the latest candidate record. A successful call commits that "
                "record and ends this review. Missing update or a verdict-record conflict returns "
                "an error and the review continues."
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
