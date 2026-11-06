"""Seed a new rollout from files and task state, without restoring conversations."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from .plan import parse_plan
from .storage import PlanStore, atomic_json_write, redact_sensitive


def locate_source(source: Path) -> tuple[Path, Path | None]:
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("Source must be an experiment, workspace, or submission directory")
    if (source / "workspace/submission").is_dir():
        return source / "workspace/submission", source
    if (source / "submission").is_dir():
        run = source.parent if source.name in {"workspace", "export"} else None
        return source / "submission", run
    if source.name == "submission" and source.parent.name in {"workspace", "export"}:
        return source, source.parent.parent
    return source, None


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_resume(
    *, paper_id: str, source: Path, experiments_root: Path,
    plan_path: Path | None = None, state_path: Path | None = None,
) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", paper_id):
        raise ValueError("Invalid paper ID")
    submission, source_run = locate_source(source)
    old_store = source_run / "agent-run/research_agent" if source_run else None
    if source_run and old_store.is_dir() and source_run.parent.name != paper_id:
        raise ValueError("Source experiment belongs to a different paper ID")
    if plan_path is None:
        candidates = [old_store / "plan.json", source_run / "input_plan.json"] if source_run else []
        plan_path = next((p for p in candidates if p.is_file()), None)
    if plan_path is None:
        raise ValueError("No plan found; provide --plan with a complete checklist plan")
    plan = parse_plan(_json(plan_path))
    source_plan = _json(old_store / "plan.json") if old_store and (old_store / "plan.json").is_file() else None
    same_plan = source_plan is not None and parse_plan(source_plan) == plan
    # Never silently apply approvals from a different plan to the new plan.
    if state_path is None and same_plan and (old_store / "plan_state.json").is_file():
        state_path = old_store / "plan_state.json"
    statuses = {t.id: "uncompleted" for t in plan.subtasks}
    if state_path is not None:
        state = _json(state_path)
        restored = state.get("statuses") if isinstance(state, dict) else None
        if not isinstance(restored, dict) or set(restored) != set(statuses):
            raise ValueError("State task IDs must match the plan exactly")
        if any(v not in {"completed", "skipped", "uncompleted"} for v in restored.values()):
            raise ValueError("Invalid task status in state file")
        statuses = restored
    # Resolve optional inherited records before allocating or copying anything.
    failures: dict[str, int] = {}
    selection = None
    feedback = None
    current = next((t.id for t in plan.subtasks if statuses[t.id] == "uncompleted"), None)
    if same_plan and state_path is not None:
        failure_path = old_store / "submission_failures.json"
        if failure_path.is_file():
            value = _json(failure_path)
            if not isinstance(value, dict) or any(
                k not in statuses or type(v) is not int or v < 0 for k, v in value.items()
            ):
                raise ValueError("Invalid inherited failure counts")
            failures = value
        selection_path = old_store / "skill_selection.json"
        if selection_path.is_file():
            selection = _json(selection_path)
        history = old_store / "tool_calls.jsonl"
        if current and history.is_file():
            for line in history.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                result = event.get("result", {})
                if (event.get("event") == "tool_result" and event.get("tool") == "submit_subtask"
                        and isinstance(result, dict) and result.get("subtask_id") == current
                        and result.get("accepted") is False):
                    feedback = result.get("message")
    output_parent = experiments_root.resolve() / paper_id
    # A destination inside the copied tree would recurse indefinitely.
    if output_parent == submission or submission in output_parent.parents:
        raise ValueError("Experiments root must not be inside the source submission")
    output_parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run = Path(tempfile.mkdtemp(prefix=f"{stamp}-ra-skill-list-resume.", dir=output_parent))
    atomic_json_write(run / "resume_origin.json", {
        "paper_id": paper_id, "source": str(source.resolve()),
        "submission_source": str(submission),
        "source_run": str(source_run) if source_run else None,
        "plan_source": str(plan_path.resolve()),
        "state_source": str(state_path.resolve()) if state_path else None,
        "restored_statuses": statuses, "conversation_restored": False,
        "created_at": datetime.now(UTC).isoformat(),
    })
    shutil.copytree(submission, run / "workspace/submission", symlinks=True)
    atomic_json_write(run / "input_plan.json", plan.model_dump())
    store = PlanStore(run / "agent-run/research_agent")
    store.initialize(plan)
    atomic_json_write(store.state_path, {"statuses": statuses})
    atomic_json_write(store.failures_path, failures)
    if selection is not None:
        atomic_json_write(store.skill_selection_path, selection)
    # No old conversations, job registry, PIDs, completion markers or checker logs.
    atomic_json_write(store.run_dir / "resume_context.json", {
        "current_task": current, "last_rejection": redact_sensitive(feedback),
        "restored_statuses": state_path is not None,
    })
    store.events.append("submission_resume_prepared", current_task=current,
                        completed_tasks=sum(v == "completed" for v in statuses.values()))
    return run


def resume_message(state_dir: Path, workspace: str) -> dict[str, str] | None:
    path = state_dir / "resume_context.json"
    if not path.is_file():
        return None
    context = _json(path)
    text = (
        "This is a new execution session continuing from an existing submission. "
        f"Existing code is in {workspace}/submission/. Inspect it before modifying it; "
        "the protected plan message contains authoritative task statuses. "
        "No previous conversation or running processes were restored. The Docker container "
        "is new: prior installed packages and environments outside the submission were not "
        "restored. Prepare necessary dependencies, obey the system's lightweight-only execution "
        "constraints, and continue the current uncompleted task. "
        "Previously completed code may still need in-scope integration fixes."
    )
    if context.get("last_rejection"):
        text += (
            "\nThe last rejection for the current task is historical feedback, not a new "
            "inspection of the copied files; verify which issues still exist:\n"
            + context["last_rejection"]
        )
    return {"role": "user", "content": text}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    try:
        run = prepare_resume(paper_id=args.paper_id, source=args.source,
                             experiments_root=args.experiments_root,
                             plan_path=args.plan, state_path=args.state)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Cannot prepare continuation: {exc}\n")
    print(run)


if __name__ == "__main__":
    main()
