"""Offline tests for the locally owned protocol, independent of any model."""
import json
import tempfile
import unittest
from pathlib import Path

from protocol.checker import parse_check_report, CheckerResponseError
from protocol.plan import parse_plan, PlanValidationError
from protocol.prompts import render_plan_message
from protocol.readonly import SafeWorkspace, PathAccessError
from protocol.state_machine import ResearchAgentTools, ResearchToolError, AgentProcess
from protocol.storage import PlanStore


PLAN = {"subtasks": [
    {"id": "task_001", "name": "First", "description": "First component"},
    {"id": "task_002", "name": "Second", "description": "Second component"},
]}


class Runtime:
    processes = []

    async def active_processes(self):
        return self.processes


class Checker:
    def __init__(self, factory):
        self.factory = factory

    async def review(self, context):
        self.factory.contexts.append(context)
        if self.factory.unavailable:
            raise RuntimeError("provider unavailable")
        accepted = self.factory.accepted
        return {"accepted": accepted, "summary": "Reviewed actual evidence",
                "issues": [] if accepted else [{"description": "Missing requirement",
                                               "evidence": "paper and submitted code"}]}


class Factory:
    accepted = False
    unavailable = False

    def __init__(self):
        self.contexts = []
        self.agents = []

    async def build(self, **kwargs):
        agent = Checker(self)
        self.agents.append(agent)
        return agent


class Snapshotter:
    async def freeze_submission(self):
        return "offline-snapshot"


class ProtocolParityTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.paper = self.workspace / "paper"
        self.paper.mkdir(parents=True)
        self.submission = self.workspace / "submission"
        self.submission.mkdir()
        (self.workspace / "instructions.txt").write_text("Implement the paper")
        self.store = PlanStore(self.root / "protocol")
        self.store.initialize(PLAN)
        self.factory = Factory()
        self.runtime = Runtime()
        self.manager = ResearchAgentTools(
            store=self.store, runtime=self.runtime, checker_factory=self.factory,
            completer_config=None, public_paper_root=self.paper,
            workspace_root=self.workspace, submission_root=self.submission,
            snapshotter=Snapshotter(),
        )

    async def submit(self, task="task_001"):
        return await self.manager.submit_subtask(
            subtask_id=task, reproduction_description="Inspect submitted source")

    async def test_skip_requires_three_real_rejections_then_confirm_final(self):
        early = await self.manager.skip_subtask(subtask_id="task_001", reason="impossible")
        self.assertIn("at least 3", early)
        for i in range(1, 4):
            self.assertIn("Check Agent rejection", await self.submit())
            self.assertEqual(self.store.failure_count("task_001"), i)
        await self.manager.skip_subtask(subtask_id="task_001", reason="impossible")
        self.factory.accepted = True
        await self.submit("task_002")
        for name in ("README.md", "reproduce.sh"):
            (self.submission / name).write_text("source")
        first = await self.manager.final_submit()
        self.assertFalse(first.accepted)
        self.assertEqual(first.errors[0]["code"], "SKIPPED_SUBTASKS_CONFIRMATION_REQUIRED")
        second = await self.manager.final_submit()
        self.assertTrue(second.accepted)
        self.assertEqual(self.manager.state.phase, "evaluation")

    async def test_fresh_checker_only_receives_current_task(self):
        await self.submit()
        await self.submit()
        self.assertIsNot(self.factory.agents[0], self.factory.agents[1])
        for context in self.factory.contexts:
            self.assertEqual(set(context.statuses), {"task_001"})
            self.assertEqual([t["id"] for t in context.plan], ["task_001"])

    async def test_failed_provider_is_not_a_rejection(self):
        self.factory.unavailable = True
        with self.assertRaises(ResearchToolError) as error:
            await self.submit()
        self.assertEqual(error.exception.code, "CHECKER_UNAVAILABLE")
        self.assertEqual(self.store.failure_count("task_001"), 0)
        self.assertEqual(self.store.status("task_001"), "uncompleted")

    async def test_pending_process_blocks_review(self):
        self.runtime.processes = [AgentProcess(pid=12345, command="offline test")]
        with self.assertRaises(ResearchToolError) as error:
            await self.submit()
        self.assertEqual(error.exception.code, "ACTIVE_AGENT_PROCESSES")
        self.assertEqual(self.factory.agents, [])

    async def test_current_task_and_plan_visibility(self):
        before = render_plan_message(self.store.render())["content"]
        self.assertIn('"id": "task_001"', before)
        self.assertNotIn('"id": "task_002"', before)
        with self.assertRaises(ResearchToolError) as error:
            await self.submit("task_002")
        self.assertEqual(error.exception.code, "NOT_CURRENT_SUBTASK")
        self.factory.accepted = True
        await self.submit()
        after = render_plan_message(self.store.render())["content"]
        self.assertIn('"id": "task_001"', after)
        self.assertIn('"id": "task_002"', after)
        self.assertEqual(self.store.failure_count("task_001"), 0)


class ContractTest(unittest.TestCase):
    def test_plan_rejects_state_and_extra_keys(self):
        for value in ({**PLAN, "status": "completed"}, {"subtasks": [
            {**PLAN["subtasks"][0], "status": "completed"}]}):
            with self.assertRaises(PlanValidationError):
                parse_plan(value)

    def test_check_report_consistency(self):
        with self.assertRaises(CheckerResponseError):
            parse_check_report({"accepted": True, "summary": "ok", "issues": [
                {"description": "wrong", "evidence": "file"}]})
        with self.assertRaises(CheckerResponseError):
            parse_check_report({"accepted": False, "summary": "wrong", "issues": []})

    def test_readonly_denies_private_and_symlink_escapes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "paper").mkdir()
            (root / "submission").mkdir()
            (root / "instructions.txt").write_text("instructions")
            (root / "paper/paper.md").write_text("public")
            (root / "submission/rubric.json").write_text("private")
            (root / "secret.txt").write_text("private")
            (root / "submission/escape").symlink_to(root / "secret.txt")
            safe = SafeWorkspace(public_root=root / "paper", workspace_root=root)
            self.assertIn("public", safe.read_file("/home/paper/paper.md"))
            for name in ("rubric.json", "escape", "../secret.txt", "/home/instructions.txt/../secret.txt"):
                with self.assertRaises(PathAccessError):
                    safe.read_file(name)


if __name__ == "__main__":
    unittest.main()
