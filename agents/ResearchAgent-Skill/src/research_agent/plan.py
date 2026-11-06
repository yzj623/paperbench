"""Frozen reproduction-plan schema and validation."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskStatus = Literal["uncompleted", "skipped", "completed"]
_TASK_ID = re.compile(r"task_[0-9]{3,}")


class PlanValidationError(ValueError):
    """A model plan did not satisfy the intentionally small JSON contract."""

    def __init__(self, details: list[str]):
        self.details = details
        super().__init__("; ".join(details))


class PlanTask(BaseModel):
    """The only fields the planning model is allowed to generate."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _TASK_ID.fullmatch(value):
            raise ValueError("id must match task_[0-9]{3,}")
        return value

    @field_validator("name", "description")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be a non-empty string")
        return value


class Plan(BaseModel):
    """A frozen plan.  Status is deliberately stored separately."""

    model_config = ConfigDict(extra="forbid")

    subtasks: list[PlanTask] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> "Plan":
        ids = [task.id for task in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("subtask ids must be unique")
        return self


class PlannedTask(PlanTask):
    """A plan item plus runtime-owned status for presentation only."""

    status: TaskStatus = "uncompleted"


def _validation_details(exc: Exception) -> list[str]:
    if hasattr(exc, "errors"):
        details: list[str] = []
        for error in exc.errors():  # type: ignore[attr-defined]
            location = ".".join(str(part) for part in error.get("loc", ())) or "plan"
            details.append(f"{location}: {error.get('msg', 'invalid value')}")
        return details
    return [str(exc)]


def parse_plan(raw: str | Mapping[str, Any]) -> Plan:
    """Parse a model response and return a strict frozen plan.

    The function intentionally rejects markdown fences, status fields, extra
    metadata, and any top-level field other than ``subtasks``.
    """

    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlanValidationError([f"invalid JSON: {exc.msg} at position {exc.pos}"]) from exc
    elif isinstance(raw, Mapping):
        value = dict(raw)
    else:
        raise PlanValidationError(["top-level value must be a JSON object"])
    if not isinstance(value, dict):
        raise PlanValidationError(["top-level value must be a JSON object"])
    try:
        return Plan.model_validate(value)
    except Exception as exc:
        raise PlanValidationError(_validation_details(exc)) from exc


def plan_with_statuses(plan: Plan, statuses: Mapping[str, TaskStatus]) -> list[PlannedTask]:
    """Render the frozen plan with runtime statuses in stable array order."""

    return [
        PlannedTask(**task.model_dump(), status=statuses[task.id]) for task in plan.subtasks
    ]


def format_invalid_plan_error(error: PlanValidationError) -> dict[str, Any]:
    """Return the structured error fed back to the planning model."""

    return {
        "error": "INVALID_PLAN",
        "details": error.details,
        "instruction": (
            "Return one corrected complete JSON object with only subtasks; "
            "do not return a patch, Markdown, or explanation."
        ),
    }
