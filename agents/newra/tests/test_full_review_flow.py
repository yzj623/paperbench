"""Exercise the full-review protocol without an LLM or Docker."""

import asyncio
import json
from pathlib import Path

from research_agent.state_machine import ResearchAgentTools, ResearchToolError
from research_agent.storage import PlanStore
from research_agent.runtime_tools import protocol_tools
from research_agent.runtime_checker import SubmitFullReviewTool
from research_agent.full_review import FullReviewReport


class Runtime:
    async def active_processes(self):
        return []


class Snapshotter:
    def __init__(self):
        self.calls = 0

    async def freeze_submission(self):
        self.calls += 1
        return "canonical.tar.gz"


class Checker:
    def __init__(self, issues):
        self.issues = issues
        self.seen = []

    async def build(self, *, completer_config, context):
        assert context.full_review
        self.seen.append(context)
        return self

    async def review(self, context):
        return {"issues": self.issues.pop(0)}


def make_manager(tmp_path: Path, issue_sets, rounds=2):
    workspace = tmp_path / "workspace"
    for part in ("paper", "submission"):
        (workspace / part).mkdir(parents=True)
    (workspace / "paper" / "paper.md").write_text("public paper")
    (workspace / "submission" / "README.md").write_text("ready")
    (workspace / "submission" / "reproduce.sh").write_text("#!/bin/sh\ntrue\n")
    (workspace / "instructions.txt").write_text("instructions")
    briefing = tmp_path / "check_instruction.txt"
    briefing.write_text("briefing")
    store = PlanStore(tmp_path / "run")
    store.initialize({"subtasks": [{"id": "task_001", "name": "implement", "description": "scope"}]})
    checker, snapshotter = Checker(issue_sets), Snapshotter()
    manager = ResearchAgentTools(
        store=store, runtime=Runtime(), checker_factory=checker, completer_config=object(),
        public_paper_root=workspace / "paper", workspace_root=workspace,
        submission_root=workspace / "submission", instructions_file=workspace / "instructions.txt",
        checker_instructions_file=briefing, snapshotter=snapshotter,
        max_review_rounds=rounds,
    )
    return manager, checker, snapshotter


ISSUE = {"issue_id": "issue_1", "description": "missing method",
         "paper_evidence": "paper section", "code_evidence": "source line"}


def test_review_response_snapshot_and_empty_review(tmp_path):
    async def exercise():
        manager, checker, canonical = make_manager(tmp_path, [[ISSUE], []])
        await manager.submit_subtask(subtask_id="task_001", reproduction_description="implemented")
        assert not checker.seen
        first = await manager.final_submit()
        assert first.accepted and manager.state.phase == "pending_review"
        assert first.snapshot_path == "canonical.tar.gz" and canonical.calls == 1
        assert Path(manager.review_store.load()["snapshots"][0]).is_file()
        feedback = await manager.run_full_review()
        assert feedback["issues"] == [ISSUE] and manager.state.phase == "review_response"
        rejected = await manager.final_submit()
        assert not rejected.accepted and rejected.errors[0]["code"] == "UNANSWERED_REVIEW_ISSUES"
        await manager.respond_to_review(issue_id="issue_1", decision="checker_inaccurate",
                                        reason="code exists", evidence="submission/file.py")
        second = await manager.final_submit()
        assert second.accepted and second.snapshot_path != first.snapshot_path
        outcome = await manager.run_full_review()
        assert outcome == {"finished": True, "reason": "empty_review", "round": 2}
        assert canonical.calls == 2 and manager.state.phase == "evaluation"
        assert len(checker.seen[1].previous_rounds) == 1
        state = manager.review_store.load()
        assert len(state["snapshots"]) == 2 and len(state["rounds"]) == 2
        assert state["rounds"][0]["responses"]["issue_1"]["decision"] == "checker_inaccurate"
    asyncio.run(exercise())


def test_round_limit_preserves_last_submission(tmp_path):
    async def exercise():
        manager, checker, canonical = make_manager(tmp_path, [[ISSUE]], rounds=1)
        await manager.submit_subtask(subtask_id="task_001", reproduction_description="done")
        await manager.final_submit()
        await manager.run_full_review()
        await manager.respond_to_review(issue_id="issue_1", decision="valid_issue",
                                        reason="fixed", evidence="submission/reproduce.sh")
        await manager.final_submit()
        outcome = await manager.run_full_review()
        assert outcome["reason"] == "review_limit" and canonical.calls == 2
        assert len(checker.seen) == 1
    asyncio.run(exercise())


def test_zero_rounds_is_no_checker(tmp_path):
    async def exercise():
        manager, checker, canonical = make_manager(tmp_path, [], rounds=0)
        await manager.submit_subtask(subtask_id="task_001", reproduction_description="done")
        await manager.final_submit()
        outcome = await manager.run_full_review()
        assert outcome["reason"] == "review_limit" and not checker.seen
        assert canonical.calls == 1
    asyncio.run(exercise())


def test_protocol_exposes_review_response_only_after_implementation(tmp_path):
    manager, _, _ = make_manager(tmp_path, [], rounds=1)
    names = {tool.name() for tool in protocol_tools(manager)}
    assert "respond_to_review" not in names
    assert "skip_subtask" in names
    manager.state.phase = "review_response"
    names = {tool.name() for tool in protocol_tools(manager)}
    assert "respond_to_review" in names
    assert "skip_subtask" not in names
    schema = SubmitFullReviewTool().get_oai_tool_call()
    assert schema["name"] == "submit_full_review"
    assert set(schema["parameters"]["properties"]["issues"]["items"]["required"]) == set(ISSUE)
    assert FullReviewReport.model_validate({"issues": [ISSUE]}).issues[0].issue_id == "issue_1"
