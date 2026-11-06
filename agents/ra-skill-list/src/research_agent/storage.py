"""Atomic, runtime-owned plan/state/event persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .plan import Plan, PlanTask, TaskStatus, parse_plan, plan_with_statuses

_SENSITIVE_KEY = re.compile(
    r"(?:api.?key|access.?key|secret|password|passwd|credential|authorization|bearer|token)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(
    r"(?:sk-[A-Za-z0-9][A-Za-z0-9_.-]*|Bearer\s+\S+)", re.IGNORECASE
)


def redact_sensitive(value: Any) -> Any:
    """Recursively remove credentials before writing ResearchAgent logs."""

    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]"
            if _SENSITIVE_KEY.search(str(key)) and str(key) not in {
                "max_tokens", "input_tokens", "output_tokens", "total_tokens",
                "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens",
                "estimated_tokens_before", "estimated_tokens_after", "token_usage",
                "token_usage_details", "prompt_tokens_details", "completion_tokens_details",
                "input_tokens_details", "output_tokens_details"
            }
            else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str):
        return _SENSITIVE_VALUE.sub("[REDACTED]", value)
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


class JsonlLog:
    """Append-only JSONL log used for tool, state, and checker events."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, **data: Any) -> None:
        import uuid
        linkage = {}
        latest = self.path.parent / "audit/latest.json"
        if latest.is_file():
            try:
                latest_value = json.loads(latest.read_text())
                linkage["snapshot_id"] = latest_value.get("snapshot_id")
                linkage["review_id"] = latest_value.get("review_id")
            except (OSError, ValueError):
                pass
        record = {
            **linkage,
            "event_id": uuid.uuid4().hex,
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **redact_sensitive(data),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def atomic_json_write(path: Path, value: Any) -> None:
    """Write a JSON document next to its destination, then atomically replace it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class PlanStore:
    """Authoritative plan/state store located beside (not inside) the workspace."""

    def __init__(self, run_dir: str | Path):
        raw_run_dir = os.fspath(run_dir)
        if "://" in raw_run_dir or raw_run_dir.startswith(("az:", "s3:", "gs:")):
            raise ValueError(
                "Research Agent PlanStore requires a local filesystem run_dir; "
                f"URI is unsupported: {raw_run_dir}"
            )
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plan_path = self.run_dir / "plan.json"
        self.state_path = self.run_dir / "plan_state.json"
        self.jobs_path = self.run_dir / "jobs.json"
        self.failures_path = self.run_dir / "submission_failures.json"
        self.skill_selection_path = self.run_dir / "skill_selection.json"
        from .audit import record
        self.events = JsonlLog(self.run_dir / "events.jsonl")
        self.tool_log = JsonlLog(self.run_dir / "tool_calls.jsonl")
        self.checker_log = JsonlLog(self.run_dir / "checker_calls.jsonl")
        self.skill_log = JsonlLog(self.run_dir / "skill_calls.jsonl")
        import hashlib
        package = Path(__file__).parent
        record(self.tool_log, "run_sources", files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in package.glob("*.py")})

    def initialize(self, plan: Plan | Mapping[str, Any] | str) -> Plan:
        if self.plan_path.exists() or self.state_path.exists():
            raise FileExistsError(f"plan already initialized in {self.run_dir}")
        parsed = plan if isinstance(plan, Plan) else parse_plan(plan)
        statuses = {task.id: "uncompleted" for task in parsed.subtasks}
        atomic_json_write(self.plan_path, parsed.model_dump())
        atomic_json_write(self.state_path, {"statuses": statuses})
        self.events.append("plan_initialized", plan_path=self.plan_path, state_path=self.state_path)
        return parsed

    def load_plan(self) -> Plan:
        return parse_plan(json.loads(self.plan_path.read_text(encoding="utf-8")))

    def statuses(self) -> dict[str, TaskStatus]:
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        statuses = data.get("statuses")
        if not isinstance(statuses, dict):
            raise ValueError("plan_state.json must contain an object named statuses")
        expected = {task.id for task in self.load_plan().subtasks}
        if set(statuses) != expected:
            raise ValueError("plan_state.json task IDs do not match the frozen plan")
        allowed = {"uncompleted", "skipped", "completed"}
        if any(value not in allowed for value in statuses.values()):
            raise ValueError("plan_state.json contains an invalid task status")
        return dict(statuses)  # type: ignore[return-value]

    def render(self) -> list[dict[str, Any]]:
        return [task.model_dump() for task in plan_with_statuses(self.load_plan(), self.statuses())]

    def status(self, task_id: str) -> TaskStatus:
        return self.statuses()[task_id]

    def first_uncompleted_task(self) -> PlanTask | None:
        """Return the first uncompleted task in frozen plan order."""

        statuses = self.statuses()
        return next(
            (task for task in self.load_plan().subtasks if statuses[task.id] == "uncompleted"),
            None,
        )

    def set_status(self, task_id: str, status: TaskStatus, *, reason: str | None = None) -> None:
        statuses = self.statuses()
        old = statuses[task_id]
        statuses[task_id] = status
        atomic_json_write(self.state_path, {"statuses": statuses})
        self.events.append(
            "task_status_changed", task_id=task_id, old_status=old, new_status=status, reason=reason
        )

    def all_completed(self) -> bool:
        return all(value == "completed" for value in self.statuses().values())

    def submission_failures(self) -> dict[str, int]:
        """Persistent per-subtask Check Agent rejection counts.

        Counts survive process restarts (keys absent from the file default to
        zero) so an invariance attack on submission retry history is not
        possible by simply restarting the solver.
        """

        if not self.failures_path.exists():
            return {}
        try:
            value = json.loads(self.failures_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(value, dict):
            return {}
        return {str(key): int(count) for key, count in value.items() if isinstance(count, int)}

    def failure_count(self, task_id: str) -> int:
        return int(self.submission_failures().get(task_id, 0))

    def record_failure(self, task_id: str, *, increment: bool) -> int:
        failures = self.submission_failures()
        if increment:
            count = failures.get(task_id, 0) + 1
        else:
            count = 0
        if count:
            failures[task_id] = count
        else:
            failures.pop(task_id, None)
        atomic_json_write(self.failures_path, failures)
        return count
