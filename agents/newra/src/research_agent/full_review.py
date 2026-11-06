"""Whole-submission review protocol and durable review history."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .storage import atomic_json_write


class FullReviewIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str
    description: str
    paper_evidence: str
    code_evidence: str

    @field_validator("issue_id", "description", "paper_evidence", "code_evidence")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be nonempty")
        return value


class FullReviewReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issues: list[FullReviewIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self) -> "FullReviewReport":
        ids = [issue.issue_id for issue in self.issues]
        if len(ids) != len(set(ids)):
            raise ValueError("issue_id must be unique within a review")
        return self


class ReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issue_id: str
    decision: Literal["plan_inaccurate", "checker_inaccurate", "valid_issue"]
    reason: str
    evidence: str

    @field_validator("issue_id", "reason", "evidence")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be nonempty")
        return value


class ReviewStore:
    def __init__(self, run_dir: Path, submission_root: Path, max_rounds: int):
        if max_rounds < 0:
            raise ValueError("max_review_rounds must be nonnegative")
        self.root = run_dir / "full_review"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.submission_root = submission_root
        self.max_rounds = max_rounds
        if not self.state_path.exists():
            atomic_json_write(self.state_path, {
                "max_rounds": max_rounds, "phase": "execution", "rounds": [], "snapshots": []
            })
        elif self.load()["max_rounds"] != max_rounds:
            raise ValueError("cannot resume with a different max_review_rounds")

    def load(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def save(self, value: dict) -> None:
        atomic_json_write(self.state_path, value)

    def freeze(self) -> str:
        """Save an inspectable source snapshot without ending the PaperBench run."""
        state = self.load()
        index = len(state["snapshots"])
        target = self.root / "snapshots" / f"submission-{index:03d}.tar.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(target, "w:gz") as archive:
            archive.add(self.submission_root, arcname="submission", recursive=True,
                        filter=lambda info: None if any(part in {".git", ".venv", "__pycache__"}
                                                 for part in Path(info.name).parts) else info)
        state["snapshots"].append(str(target))
        self.save(state)
        return str(target)
