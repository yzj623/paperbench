"""Stateless Check Agent protocol and strict response parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .plan import PlanTask


class CheckIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    evidence: str

    @field_validator("description", "evidence")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class CheckReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: StrictBool
    summary: str
    issues: list[CheckIssue] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def summary_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must be non-empty")
        return value

    @model_validator(mode="after")
    def accepted_issue_consistency(self) -> "CheckReport":
        if self.accepted is True and self.issues:
            raise ValueError("accepted=true requires issues=[]")
        if self.accepted is False and not self.issues:
            raise ValueError("accepted=false requires at least one issue")
        return self


class CheckerResponseError(ValueError):
    """Malformed or schema-invalid Check Agent response."""

    def __init__(self, details: list[str]):
        self.details = details
        super().__init__("; ".join(details))


def parse_check_report(raw: str | Mapping[str, Any]) -> CheckReport:
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CheckerResponseError([f"invalid JSON: {exc.msg} at position {exc.pos}"]) from exc
    elif isinstance(raw, Mapping):
        value = dict(raw)
    else:
        raise CheckerResponseError(["response must be a JSON object"])
    if not isinstance(value, dict):
        raise CheckerResponseError(["response must be a JSON object"])
    try:
        return CheckReport.model_validate(value)
    except Exception as exc:
        details = []
        if hasattr(exc, "errors"):
            details = [
                f"{'.'.join(str(part) for part in item.get('loc', ())) or 'report'}: "
                f"{item.get('msg', 'invalid value')}"
                for item in exc.errors()  # type: ignore[attr-defined]
            ]
        else:
            details = [str(exc)]
        raise CheckerResponseError(details) from exc


@dataclass(frozen=True)
class CheckAgentContext:
    task: PlanTask
    # The state-machine supplies only the current task and its runtime-owned
    # status. ``Any`` keeps this context usable by
    # adapters that represent plan entries as JSON mappings.
    plan: Sequence[Any]
    statuses: Mapping[str, str]
    reproduction_description: str
    read_only_tools: Sequence[Any]
    read_only_dispatcher: Any | None = None
    tool_log: Any | None = None
    full_review: bool = False
    previous_rounds: Sequence[Any] = ()


class StatelessChecker(Protocol):
    async def review(self, context: CheckAgentContext) -> str | Mapping[str, Any]: ...


class CheckerFactory(Protocol):
    async def build(self, *, completer_config: Any, context: CheckAgentContext) -> StatelessChecker: ...


async def review_with_retries(
    factory: CheckerFactory,
    context: CheckAgentContext,
    *,
    completer_config: Any,
    log: Any | None = None,
    max_retries: int = 3,
) -> CheckReport:
    """Build a fresh checker for every attempt and retry only bad responses."""

    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    last_error: CheckerResponseError | None = None
    for attempt in range(1, max_retries + 1):
        try:
            checker = await factory.build(completer_config=completer_config, context=context)
            if log is not None:
                log.append("checker_started", attempt=attempt, task_id=context.task.id)
            raw = await checker.review(context)
            report = parse_check_report(raw)
        except CheckerResponseError as exc:
            last_error = exc
            if log is not None:
                log.append(
                    "checker_response_invalid",
                    attempt=attempt,
                    task_id=context.task.id,
                    details=exc.details,
                )
            continue
        except Exception as exc:  # provider/network/protocol errors are retryable too
            if log is not None:
                log.append(
                    "checker_call_error",
                    attempt=attempt,
                    task_id=context.task.id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            continue
        if log is not None:
            log.append(
                "checker_response",
                attempt=attempt,
                task_id=context.task.id,
                report=report.model_dump(),
            )
        return report
    if last_error is not None:
        raise last_error
    raise CheckerResponseError(["Check Agent call failed on every retry"])
