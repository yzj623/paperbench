"""Stateless Check Agent protocol and strict response parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


CheckStatus = Literal[
    "correct", "equivalent", "approximate", "incorrect", "checklist_error"
]


class CheckItemReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: CheckStatus
    reason: str
    evidence: str

    @field_validator("id", "reason", "evidence")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class ChecklistReviewSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviews: list[CheckItemReview]


class UnlistedIssuesSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issues: list[CheckIssue]


class CheckReport(BaseModel):
    """Raw checker findings; acceptance is computed, never submitted by the model."""

    model_config = ConfigDict(extra="forbid")

    reviews: list[CheckItemReview]
    unlisted_issues: list[CheckIssue]

    @property
    def accepted(self) -> bool:
        return not self.unlisted_issues and all(
            review.status in {"correct", "equivalent", "checklist_error"}
            for review in self.reviews
        )

    @property
    def issues(self) -> list[CheckIssue]:
        issues = [
            CheckIssue(
                description=f"Checklist {review.id} ({review.status}): {review.reason}",
                evidence=review.evidence,
            )
            for review in self.reviews
            if review.status in {"approximate", "incorrect"}
        ]
        return [*issues, *self.unlisted_issues]

    @property
    def summary(self) -> str:
        return "Checklist and independent review passed" if self.accepted else "Checker found fidelity issues"


class CheckerResponseError(ValueError):
    """Malformed or schema-invalid Check Agent response."""

    def __init__(self, details: list[str]):
        self.details = details
        super().__init__("; ".join(details))


def _parse_model(raw: str | Mapping[str, Any], model: type[BaseModel]) -> BaseModel:
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
        return model.model_validate(value)
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


def parse_checklist_reviews(
    raw: str | Mapping[str, Any], *, expected_ids: Sequence[str]
) -> ChecklistReviewSubmission:
    submission = _parse_model(raw, ChecklistReviewSubmission)
    assert isinstance(submission, ChecklistReviewSubmission)
    actual = [review.id for review in submission.reviews]
    expected = set(expected_ids)
    missing = sorted(expected - set(actual))
    extra = sorted(set(actual) - expected)
    duplicates = sorted({item for item in actual if actual.count(item) > 1})
    if missing or extra or duplicates:
        raise CheckerResponseError(
            [f"checklist IDs must match exactly; missing={missing}, extra={extra}, duplicates={duplicates}"]
        )
    return submission


def parse_unlisted_issues(raw: str | Mapping[str, Any]) -> UnlistedIssuesSubmission:
    submission = _parse_model(raw, UnlistedIssuesSubmission)
    assert isinstance(submission, UnlistedIssuesSubmission)
    return submission


def parse_check_report(
    raw: str | Mapping[str, Any], *, expected_ids: Sequence[str] | None = None
) -> CheckReport:
    report = _parse_model(raw, CheckReport)
    assert isinstance(report, CheckReport)
    if expected_ids is not None:
        parse_checklist_reviews({"reviews": [item.model_dump() for item in report.reviews]}, expected_ids=expected_ids)
    return report


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
            report = parse_check_report(raw, expected_ids=[check.id for check in context.task.checklist])
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
