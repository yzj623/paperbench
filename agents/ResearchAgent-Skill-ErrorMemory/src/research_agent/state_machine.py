"""State machine and tool protocol for the planner/checker agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .checker import CheckAgentContext, CheckerFactory, review_with_retries
from .readonly import (
    PathAccessError,
    ReadOnlyToolDispatcher,
    SafeWorkspace,
    readonly_tool_schemas,
)
from .storage import PlanStore


@dataclass(frozen=True)
class AgentProcess:
    pid: int
    command: str


# After this many consecutive Check Agent rejections for one subtask, the
# submit_subtask tool result appends an explicit hint that the agent may
# reasonably consider using skip_subtask for a genuinely impossible task.
_SUBTASK_FAILURE_HINT_THRESHOLD = 3


class ResearchToolRuntime(Protocol):
    async def active_processes(self) -> Sequence[AgentProcess]: ...


class Snapshotter(Protocol):
    async def freeze_submission(self) -> str: ...


class ResearchToolError(RuntimeError):
    """A recoverable tool-level error; authoritative state is unchanged."""

    def __init__(self, code: str, message: str, **details: Any):
        self.code = code
        self.details = details
        super().__init__(message)


@dataclass(frozen=True)
class FinalSubmitResult:
    accepted: bool
    snapshot_path: str | None = None
    errors: tuple[dict[str, Any], ...] = ()


@dataclass
class ResearchAgentState:
    phase: str = "execution"
    # A second final_submit call may explicitly confirm proceeding with
    # skipped subtasks, but only after an otherwise-valid call was rejected
    # solely for that reason. The confirmation is deliberately not bound to
    # a particular set of skipped task IDs. Other failures never arm it.
    skipped_submission_confirmation_pending: bool = False


class ResearchAgentTools:
    """Implementation of skip_subtask, submit_subtask, and final_submit."""

    def __init__(
        self,
        *,
        store: PlanStore,
        runtime: ResearchToolRuntime,
        checker_factory: CheckerFactory,
        completer_config: Any,
        public_paper_root: str | Path,
        workspace_root: str | Path,
        submission_root: str | Path | None = None,
        instructions_file: str | Path | None = None,
        snapshotter: Snapshotter | None = None,
        job_manager: Any | None = None,
        checker_instructions_file: str | Path | None = None,
        required_files: Sequence[str] = ("README.md", "reproduce.sh"),
    ):
        self.store = store
        self.runtime = runtime
        self.checker_factory = checker_factory
        self.completer_config = completer_config
        self.readonly_workspace = SafeWorkspace(
            public_root=public_paper_root,
            workspace_root=workspace_root,
            submission_root=submission_root,
            instructions_file=instructions_file,
            checker_instructions_file=checker_instructions_file,
        )
        self.snapshotter = snapshotter
        self.job_manager = job_manager
        self.required_files = tuple(required_files)
        self.state = ResearchAgentState()

    async def _active_processes(self) -> list[AgentProcess]:
        """Merge the runtime scan and managed-job registry without PID duplicates."""

        # The runtime scan catches shell-launched descendants (including
        # legacy commands), while the local registry catches a managed job
        # during the small interval before /proc reflects its token.  Query
        # both every time so a submission cannot race a newly started job.
        values: list[AgentProcess] = list(await self.runtime.active_processes())
        if self.job_manager is not None:
            values.extend(await self.job_manager.active_processes())
        by_pid: dict[int, AgentProcess] = {}
        for item in values:
            by_pid.setdefault(int(item.pid), item)
        return list(by_pid.values())

    def _require_execution_phase(self) -> None:
        if self.state.phase != "execution":
            raise ResearchToolError(
                "INVALID_STATE", f"Agent phase is already {self.state.phase}; tools are closed."
            )

    def _task(self, task_id: str) -> Any:
        for task in self.store.load_plan().subtasks:
            if task.id == task_id:
                return task
        raise ResearchToolError("UNKNOWN_SUBTASK", f"Unknown subtask id: {task_id}", subtask_id=task_id)

    def _log_call(self, tool: str, arguments: Mapping[str, Any]) -> None:
        from .audit import snapshot
        import uuid

        self._audit_review_id = uuid.uuid4().hex
        snapshot(
            self.store,
            self.readonly_workspace.submission_root,
            "before",
            tool,
            {**dict(arguments), "review_id": self._audit_review_id},
        )
        self.store.tool_log.append("tool_call", tool=tool, arguments=dict(arguments))

    def _require_current_task(self, task_id: str) -> None:
        current = self.store.first_uncompleted_task()
        if current is None or current.id != task_id:
            raise ResearchToolError(
                "NOT_CURRENT_SUBTASK",
                "Only the current uncompleted subtask may be submitted or skipped.",
                current_subtask_id=current.id if current else None,
            )

    def _log_result(self, tool: str, result: Mapping[str, Any]) -> None:
        from .audit import snapshot

        snapshot(
            self.store,
            self.readonly_workspace.submission_root,
            "after",
            tool,
            {**dict(result), "review_id": getattr(self, "_audit_review_id", None)},
        )
        self.store.tool_log.append("tool_result", tool=tool, result=dict(result))

    async def skip_subtask(self, *, subtask_id: str, reason: str) -> str:
        self._require_execution_phase()
        self._log_call("skip_subtask", {"subtask_id": subtask_id, "reason": reason})
        if not isinstance(reason, str) or not reason.strip():
            raise ResearchToolError("INVALID_ARGUMENT", "reason must be a non-empty string")
        self._task(subtask_id)
        self._require_current_task(subtask_id)
        current = self.store.status(subtask_id)
        if current != "uncompleted":
            raise ResearchToolError(
                "INVALID_STATE",
                f"skip_subtask requires uncompleted; current state is {current}",
                subtask_id=subtask_id,
            )
        failures = self.store.failure_count(subtask_id)
        if failures < _SUBTASK_FAILURE_HINT_THRESHOLD:
            result = (
                f"Please continue trying to complete subtask {subtask_id}. It has received "
                f"{failures} valid Check Agent rejection(s); skip_subtask is allowed only after "
                f"at least {_SUBTASK_FAILURE_HINT_THRESHOLD} valid Check Agent rejections. "
                "The subtask remains uncompleted."
            )
            self.store.events.append(
                "skip_blocked_insufficient_failures",
                task_id=subtask_id,
                failure_count=failures,
                required_failures=_SUBTASK_FAILURE_HINT_THRESHOLD,
            )
            self._log_result("skip_subtask", {"ok": False, "message": result})
            return result
        self.store.set_status(subtask_id, "skipped", reason=reason)
        result = f"The corresponding task in the reproduction plan has been changed to skipped: {subtask_id}."
        self._log_result("skip_subtask", {"ok": True, "message": result})
        return result

    async def submit_subtask(
        self, *, subtask_id: str, reproduction_description: str
    ) -> str:
        self._require_execution_phase()
        self._log_call(
            "submit_subtask",
            {
                "subtask_id": subtask_id,
                "reproduction_description": reproduction_description,
            },
        )
        if not isinstance(reproduction_description, str) or not reproduction_description.strip():
            raise ResearchToolError(
                "INVALID_ARGUMENT", "reproduction_description must be a non-empty string"
            )
        task = self._task(subtask_id)
        self._require_current_task(subtask_id)
        current = self.store.status(subtask_id)
        if current != "uncompleted":
            raise ResearchToolError(
                "INVALID_STATE",
                f"submit_subtask requires uncompleted; current state is {current}",
                subtask_id=subtask_id,
            )
        active = await self._active_processes()
        if active:
            details = [{"pid": item.pid, "command": item.command} for item in active]
            self.store.events.append("submit_blocked_active_processes", task_id=subtask_id, processes=details)
            raise ResearchToolError(
                "ACTIVE_AGENT_PROCESSES",
                "All processes and descendants started by the agent must end before submission.",
                processes=details,
            )

        active_error_record = self.store.error_record(subtask_id)
        self.store.checker_log.append(
            "checker_error_record_loaded",
            task_id=subtask_id,
            issues=active_error_record,
        )
        context = CheckAgentContext(
            task=task,
            plan=[{**task.model_dump(), "status": current}],
            statuses={task.id: current},
            reproduction_description=reproduction_description,
            read_only_tools=readonly_tool_schemas(),
            error_record=active_error_record,
            error_record_store=self.store,
            read_only_dispatcher=ReadOnlyToolDispatcher(self.readonly_workspace),
            tool_log=self.store.checker_log,
        )
        try:
            report = await review_with_retries(
                self.checker_factory,
                context,
                completer_config=self.completer_config,
                log=self.store.checker_log,
            )
        except Exception as exc:
            # A checker/provider failure is neither acceptance nor rejection;
            # leave the authoritative status unchanged for a later submission.
            self.store.events.append(
                "checker_unavailable", task_id=subtask_id, error=f"{type(exc).__name__}: {exc}"
            )
            raise ResearchToolError(
                "CHECKER_UNAVAILABLE",
                "Check Agent did not return a valid report after the allowed retries.",
                details=str(exc),
            ) from exc
        if report.accepted:
            self.store.set_status(subtask_id, "completed", reason=report.summary)
            self.store.record_failure(subtask_id, increment=False)
            result = f"The corresponding task in the reproduction plan has been changed to completed: {subtask_id}."
        else:
            self.store.set_status(subtask_id, "uncompleted", reason=report.summary)
            failures = self.store.record_failure(subtask_id, increment=True)
            result = (
                f"The corresponding task in the reproduction plan has been changed to uncompleted: "
                f"{subtask_id}. Check Agent rejection: "
                + " ".join(f"{issue.description} Evidence: {issue.evidence}" for issue in report.issues)
            )
            if failures >= _SUBTASK_FAILURE_HINT_THRESHOLD:
                result += (
                    f" You have now failed this subtask {failures} times. If you believe these "
                    "rejections reflect a task that is theoretically impossible to complete with "
                    "the provided materials, rather than an unfaithful reproduction of the paper, "
                    "you may use the skip_subtask tool to defer its check and continue with later "
                    "work."
                )
        self._log_result(
            "submit_subtask",
            {"ok": True, "subtask_id": subtask_id, "accepted": report.accepted, "message": result},
        )
        return result

    async def final_submit(self) -> FinalSubmitResult:
        self._require_execution_phase()
        self._log_call("final_submit", {})
        errors: list[dict[str, Any]] = []
        statuses = self.store.statuses()
        uncompleted = [task_id for task_id, status in statuses.items() if status == "uncompleted"]
        skipped = [task_id for task_id, status in statuses.items() if status == "skipped"]
        if uncompleted:
            errors.append(
                {
                    "code": "UNCOMPLETED_SUBTASKS",
                    "message": (
                        "Every subtask must be completed or validly skipped before final "
                        "submission. Continue working on all uncompleted subtasks. This "
                        "rejection does not count as skipped-work confirmation."
                    ),
                    "subtask_ids": uncompleted,
                }
            )
            result = FinalSubmitResult(accepted=False, errors=tuple(errors))
            self._log_result("final_submit", {"ok": False, "errors": errors})
            return result
        active = await self._active_processes()
        if active:
            errors.append(
                {
                    "code": "ACTIVE_AGENT_PROCESSES",
                    "message": "Agent-started processes remain active.",
                    "processes": [{"pid": item.pid, "command": item.command} for item in active],
                }
            )
        for relative in self.required_files:
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts:
                errors.append(
                    {"code": "INVALID_REQUIRED_FILE", "message": f"Invalid required path: {relative}"}
                )
                continue
            try:
                raw_target = self.readonly_workspace.submission_root / candidate
                if raw_target.is_symlink():
                    raise PathAccessError("required file cannot be a symlink")
                target = self.readonly_workspace.resolve(
                    self.readonly_workspace.submission_root / candidate
                )
            except (PathAccessError, OSError):
                target = None
            if (
                target is None
                or target.is_symlink()
                or not target.is_file()
                or target.stat().st_size == 0
            ):
                errors.append(
                    {
                        "code": "MISSING_REQUIRED_FILE",
                        "message": f"Required non-empty regular file is missing: {relative}",
                    }
                )
        if self.snapshotter is None:
            errors.append(
                {
                    "code": "SNAPSHOTTER_UNAVAILABLE",
                    "message": "A submission snapshotter is required before entering evaluation.",
                }
            )
        # Only an otherwise-valid submission may consume the one skipped-work
        # confirmation attempt. Uncompleted tasks and every hard validation
        # failure return directly without creating or consuming confirmation.
        if (
            not errors
            and skipped
            and not self.state.skipped_submission_confirmation_pending
        ):
            errors.append(
                {
                    "code": "SKIPPED_SUBTASKS_CONFIRMATION_REQUIRED",
                    "message": (
                        "All remaining non-completed subtasks are validly skipped. Call "
                        "final_submit again to explicitly confirm entering evaluation "
                        "with skipped work."
                    ),
                    "subtask_ids": skipped,
                }
            )
        if errors:
            if len(errors) == 1 and errors[0].get("code") == (
                "SKIPPED_SUBTASKS_CONFIRMATION_REQUIRED"
            ):
                self.state.skipped_submission_confirmation_pending = True
                self.store.events.append(
                    "final_submit_skipped_confirmation_required",
                    subtask_ids=skipped,
                )
            result = FinalSubmitResult(accepted=False, errors=tuple(errors))
            self._log_result("final_submit", {"ok": False, "errors": errors})
            return result
        if skipped and self.state.skipped_submission_confirmation_pending:
            self.store.events.append(
                "final_submit_skipped_confirmed",
                subtask_ids=skipped,
            )
        snapshot_path = None
        try:
            snapshot_path = await self.snapshotter.freeze_submission()
        except Exception as exc:
            error = {
                "code": "SNAPSHOT_FAILED",
                "message": f"Could not freeze the submission: {type(exc).__name__}: {exc}",
            }
            self.store.events.append("snapshot_failed", error=error)
            result = FinalSubmitResult(accepted=False, errors=(error,))
            self._log_result("final_submit", {"ok": False, "errors": [error]})
            return result
        self.state.phase = "evaluation"
        self.store.events.append("agent_phase_finished", snapshot_path=snapshot_path)
        result = FinalSubmitResult(accepted=True, snapshot_path=snapshot_path)
        self._log_result(
            "final_submit", {"ok": True, "snapshot_path": snapshot_path, "phase": "evaluation"}
        )
        return result

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "skip_subtask",
                    "description": (
                        "Temporarily defer a genuinely impossible subtask only after it has received "
                        "at least three valid Check Agent rejections. Earlier calls leave it uncompleted."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subtask_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["subtask_id", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_subtask",
                    "description": "Submit one subtask to a fresh stateless Check Agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subtask_id": {"type": "string"},
                            "reproduction_description": {"type": "string"},
                        },
                        "required": ["subtask_id", "reproduction_description"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "final_submit",
                    "description": (
                        "Finish only when every subtask is completed or validly skipped and all "
                        "hard gates pass. Uncompleted tasks are always rejected. If skipped tasks "
                        "remain, the first otherwise-valid call asks for confirmation and a later "
                        "call confirms evaluation with skipped work."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
        ]
