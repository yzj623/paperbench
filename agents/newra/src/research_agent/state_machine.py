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
from .full_review import FullReviewReport, ReviewResponse, ReviewStore


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
        max_review_rounds: int | None = None,
    ):
        self.store = store
        self.runtime = runtime
        self.checker_factory = checker_factory
        self.completer_config = completer_config
        self.checker_instructions_file = checker_instructions_file
        self.readonly_workspace = SafeWorkspace(
            public_root=public_paper_root,
            workspace_root=workspace_root,
            submission_root=submission_root,
            instructions_file=instructions_file,
            checker_instructions_file=(checker_instructions_file
                                       if max_review_rounds is None else None),
        )
        self.snapshotter = snapshotter
        self.job_manager = job_manager
        self.required_files = tuple(required_files)
        self.state = ResearchAgentState()
        self.review_store = (ReviewStore(store.run_dir, self.readonly_workspace.submission_root,
                                         max_review_rounds) if max_review_rounds is not None else None)
        if self.review_store is not None:
            self.state.phase = self.review_store.load()["phase"]

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
        snapshot(self.store, self.readonly_workspace.submission_root, "before", tool,
                 {**dict(arguments), "review_id": self._audit_review_id})
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
        snapshot(self.store, self.readonly_workspace.submission_root, "after", tool,
                 {**dict(result), "review_id": getattr(self, "_audit_review_id", None)})
        self.store.tool_log.append("tool_result", tool=tool, result=dict(result))

    async def skip_subtask(self, *, subtask_id: str, reason: str) -> str:
        if self.review_store is not None and self.state.phase != "execution":
            raise ResearchToolError("UNAVAILABLE", "skip_subtask is unavailable in full-review mode")
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

        if self.review_store is not None:
            self.store.set_status(
                subtask_id, "completed",
                reason="Accepted automatically because the checker is disabled.",
            )
            self.store.record_failure(subtask_id, increment=False)
            result = (
                "The corresponding task in the reproduction plan has been changed to completed: "
                f"{subtask_id}."
            )
            self._log_result("submit_subtask", {"ok": True, "subtask_id": subtask_id,
                                                "accepted": True, "message": result})
            return result

        context = CheckAgentContext(
            task=task,
            plan=[{**task.model_dump(), "status": current}],
            statuses={task.id: current},
            reproduction_description=reproduction_description,
            read_only_tools=readonly_tool_schemas(),
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
        if self.review_store is None:
            self._require_execution_phase()
        elif self.state.phase not in {"execution", "review_response"}:
            raise ResearchToolError("INVALID_STATE", f"final_submit unavailable in {self.state.phase}")
        self._log_call("final_submit", {})
        errors: list[dict[str, Any]] = []
        if self.review_store is not None and self.state.phase == "review_response":
            current = self.review_store.load()["rounds"][-1]
            missing = [item["issue_id"] for item in current["issues"]
                       if item["issue_id"] not in current["responses"]]
            if missing:
                errors.append({"code": "UNANSWERED_REVIEW_ISSUES",
                               "message": "Respond to every review issue before final_submit.",
                               "issue_ids": missing})
                self._log_result("final_submit", {"ok": False, "errors": errors})
                return FinalSubmitResult(accepted=False, errors=tuple(errors))
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
            (self.review_store is None or self.state.phase == "execution") and
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
        if self.review_store is not None:
            try:
                # Phase one uses the same canonical snapshotter and tool result
                # as standalone NoChecker. A separate review snapshot preserves
                # the initial code while the same main-agent session continues.
                canonical_path = None
                if self.state.phase == "execution":
                    freeze = self.snapshotter.freeze_submission
                    if self.review_store.max_rounds:
                        freeze = getattr(self.snapshotter, "freeze_for_review", freeze)
                    canonical_path = await freeze()
                snapshot_path = self.review_store.freeze()
            except Exception as exc:
                error = {"code": "SNAPSHOT_FAILED",
                         "message": f"Could not freeze the submission: {type(exc).__name__}: {exc}"}
                self.store.events.append("snapshot_failed", error=error)
                self._log_result("final_submit", {"ok": False, "errors": [error]})
                return FinalSubmitResult(accepted=False, errors=(error,))
            review_state = self.review_store.load()
            review_state["phase"] = "pending_review"
            self.review_store.save(review_state)
            self.state.phase = "pending_review"
            self.store.events.append("full_review_snapshot", snapshot_path=snapshot_path,
                                     snapshot_number=len(review_state["snapshots"]))
            self._log_result("final_submit", {"ok": True, "snapshot_path": snapshot_path,
                                               "phase": "pending_review"})
            return FinalSubmitResult(accepted=True, snapshot_path=canonical_path or snapshot_path)

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

    async def respond_to_review(self, *, issue_id: str, decision: str,
                                reason: str, evidence: str) -> str:
        if self.review_store is None or self.state.phase != "review_response":
            raise ResearchToolError("INVALID_STATE", "No pending review requires a response")
        response = ReviewResponse.model_validate({"issue_id": issue_id, "decision": decision,
                                                  "reason": reason, "evidence": evidence})
        self._log_call("respond_to_review", response.model_dump())
        state = self.review_store.load()
        current = state["rounds"][-1]
        if issue_id not in {item["issue_id"] for item in current["issues"]}:
            raise ResearchToolError("UNKNOWN_ISSUE", f"Unknown issue_id: {issue_id}")
        if issue_id in current["responses"]:
            raise ResearchToolError("ALREADY_ANSWERED", f"Issue already answered: {issue_id}")
        current["responses"][issue_id] = response.model_dump()
        self.review_store.save(state)
        self.store.events.append("full_review_response", **response.model_dump())
        self._log_result("respond_to_review", {"ok": True, "issue_id": issue_id})
        return f"Response recorded for {issue_id}."

    async def run_full_review(self) -> dict:
        if self.review_store is None or self.state.phase != "pending_review":
            raise RuntimeError("No full review is pending")
        state = self.review_store.load()
        if len(state["rounds"]) >= self.review_store.max_rounds:
            if state["rounds"]:
                await self.snapshotter.freeze_submission()
            state["phase"] = "evaluation"
            self.review_store.save(state)
            self.state.phase = "evaluation"
            return {"finished": True, "reason": "review_limit"}

        import copy
        checker_workspace = SafeWorkspace(
            public_root=self.readonly_workspace.public_root,
            workspace_root=self.readonly_workspace.workspace_root,
            submission_root=self.readonly_workspace.submission_root,
            instructions_file=self.readonly_workspace.instructions_file,
            checker_instructions_file=self.checker_instructions_file,
        )
        context = CheckAgentContext(
            task=self.store.load_plan().subtasks[0], plan=self.store.render(),
            statuses=self.store.statuses(), reproduction_description="full submission",
            read_only_tools=readonly_tool_schemas(),
            read_only_dispatcher=ReadOnlyToolDispatcher(checker_workspace),
            tool_log=self.store.checker_log, full_review=True,
            previous_rounds=copy.deepcopy(state["rounds"]),
        )
        # A new checker is constructed for each review, and again on a malformed
        # response or provider failure. A failed check never consumes a round.
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                checker = await self.checker_factory.build(completer_config=self.completer_config,
                                                           context=context)
                raw = await checker.review(context)
                report = FullReviewReport.model_validate(raw)
                break
            except Exception as exc:
                last_error = exc
                self.store.checker_log.append("full_review_retry", attempt=attempt,
                                              error=f"{type(exc).__name__}: {exc}")
        else:
            raise RuntimeError("Full review failed after three attempts") from last_error
        number = len(state["rounds"]) + 1
        state["rounds"].append({"number": number,
                                "snapshot_path": state["snapshots"][-1],
                                "issues": [item.model_dump() for item in report.issues],
                                "responses": {}})
        if not report.issues:
            await self.snapshotter.freeze_submission()
            state["phase"] = "evaluation"
            self.review_store.save(state)
            self.state.phase = "evaluation"
            return {"finished": True, "reason": "empty_review", "round": number}
        state["phase"] = "review_response"
        self.review_store.save(state)
        self.state.phase = "review_response"
        return {"finished": False, "round": number, "issues": state["rounds"][-1]["issues"],
                "snapshot_path": state["snapshots"][-1]}

    def tool_schemas(self) -> list[dict[str, Any]]:
        if self.review_store is not None and self.state.phase != "execution":
            from .runtime_tools import protocol_tools
            names = {"submit_subtask", "final_submit", "respond_to_review"}
            return [{"type": "function", "function": {
                "name": schema["name"], "description": schema["description"],
                "parameters": schema["parameters"],
            }} for tool in protocol_tools(self) if tool.name() in names
                for schema in [tool.get_oai_tool_call()]]
        return [
            {
                "type": "function",
                "function": {
                    "name": "skip_subtask",
                    "description": (
                        "Legacy compatibility tool. Its rejection-count gate cannot normally be "
                        "reached when subtask submissions are accepted directly."
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
                    "description": (
                        "Submit the current subtask and mark it completed directly after protocol "
                        "validation succeeds."
                    ),
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
