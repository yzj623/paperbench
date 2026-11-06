import asyncio
import json
from types import SimpleNamespace

from research_agent.context import MessageWindow
from research_agent.runtime_checker import ErrorRecordTool
from research_agent.plan import PlanTask
from research_agent.runtime_checker import RuntimeChecker
from research_agent.storage import PlanStore


def test_error_record_update_reindexes_and_replaces_candidate(tmp_path):
    store = PlanStore(tmp_path / "run")
    initial = [
        {"issue_id": 1, "name": "first", "description": "first description"},
        {"issue_id": 2, "name": "second", "description": "second description"},
    ]
    tool = ErrorRecordTool(original_record=initial, log=store.checker_log)

    first_result = json.loads(
        asyncio.run(
            tool.execute(
                None,
                existing_issue_actions=[
                    {"issue_id": 1, "action": "keep", "reason": "still present"},
                    {"issue_id": 2, "action": "delete", "reason": "verified fixed"},
                ],
                new_issues=[{"name": "third", "description": "new defect"}],
            )
        )
    )
    assert first_result["candidate_error_record"] == [
        {"issue_id": 1, "name": "first", "description": "first description"},
        {"issue_id": 2, "name": "third", "description": "new defect"},
    ]

    second_result = json.loads(
        asyncio.run(
            tool.execute(
                None,
                existing_issue_actions=[
                    {
                        "issue_id": 1,
                        "action": "keep",
                        "reason": "still present",
                        "new_name": "renamed first",
                    },
                    {"issue_id": 2, "action": "delete", "reason": "verified fixed"},
                ],
                new_issues=[],
            )
        )
    )
    assert second_result["superseded_previous_candidate"] is True
    assert second_result["candidate_error_record"] == [
        {"issue_id": 1, "name": "renamed first", "description": "first description"}
    ]
    assert tool.success_count == 2
    assert store.checker_log.path.exists()


def test_error_record_update_rejects_missing_or_duplicate_original_ids():
    original = [
        {"issue_id": 1, "name": "first", "description": "first description"},
        {"issue_id": 2, "name": "second", "description": "second description"},
    ]
    tool = ErrorRecordTool(original_record=original)
    result = json.loads(
        asyncio.run(
            tool.execute(
                None,
                existing_issue_actions=[
                    {"issue_id": 1, "action": "keep", "reason": "still present"}
                ],
                new_issues=[],
            )
        )
    )
    assert result["ok"] is False
    assert tool.success_count == 0


def test_committed_error_record_is_per_task_and_persistent(tmp_path):
    store = PlanStore(tmp_path / "run")
    record = [{"issue_id": 1, "name": "bug", "description": "specific defect"}]
    store.commit_error_record("task_001", record, reason="rejected")

    restored = PlanStore(store.run_dir)
    assert restored.error_record("task_001") == record
    assert restored.error_record("task_002") == []
    saved = json.loads(restored.error_records_path.read_text(encoding="utf-8"))
    assert saved["tasks"]["task_001"]["issues"] == record


def test_protected_error_record_survives_history_trimming():
    window = MessageWindow(
        {"role": "system", "content": "system"},
        {"role": "user", "content": "current task"},
    )
    window.update_error_record('{"issue_id":1,"name":"bug"}')
    window.append({"role": "user", "content": "old history"})
    window.append({"role": "assistant", "content": "new history"})
    window.trim_for_context(0)

    messages = window.messages
    assert [message["content"] for message in messages] == [
        "system",
        "current task",
        '{"issue_id":1,"name":"bug"}',
    ]


def test_runtime_checker_commits_only_after_consistent_review(tmp_path, monkeypatch):
    store = PlanStore(tmp_path / "run")
    task = PlanTask(id="task_001", name="implement", description="implement method")
    dispatcher = SimpleNamespace(schemas={}, workspace=SimpleNamespace(workspace_root=tmp_path))
    context = SimpleNamespace(
        task=task,
        plan=[{**task.model_dump(), "status": "uncompleted"}],
        statuses={"task_001": "uncompleted"},
        reproduction_description="inspect method implementation",
        read_only_tools=[],
        error_record=[],
        error_record_store=store,
        read_only_dispatcher=dispatcher,
        tool_log=store.checker_log,
    )

    class FakeSettings:
        def new_config(self, **kwargs):
            self.tools = kwargs["basicagent_tools"]
            return self

        def build(self):
            return SimpleNamespace(context_window=None)

    update = SimpleNamespace(
        name="update_error_record",
        arguments={
            "existing_issue_actions": [],
            "new_issues": [{"name": "missing method", "description": "required function absent"}],
        },
        call_id="update-1",
    )
    premature_accept = SimpleNamespace(
        name="submit_review",
        arguments={"accepted": True, "summary": "looks complete", "issues": []},
        call_id="submit-1",
    )
    final_reject = SimpleNamespace(
        name="submit_review",
        arguments={
            "accepted": False,
            "summary": "required method missing",
            "issues": [{"description": "missing method", "evidence": "file does not define it"}],
        },
        call_id="submit-2",
    )
    responses = iter(
        [
            SimpleNamespace(response_messages=[], tool_calls=[update, premature_accept]),
            SimpleNamespace(response_messages=[], tool_calls=[final_reject]),
        ]
    )

    async def fake_request(**kwargs):
        assert "update_error_record" in {tool.name() for tool in checker._tools()}
        return next(responses)

    monkeypatch.setattr(
        "research_agent.runtime_checker.make_completer_request", fake_request
    )
    checker = RuntimeChecker(completer_config=FakeSettings(), context=context, max_steps=4)
    report = asyncio.run(checker.review(context))

    assert report["accepted"] is False
    assert store.error_record("task_001") == [
        {"issue_id": 1, "name": "missing method", "description": "required function absent"}
    ]
    events = [json.loads(line)["event"] for line in store.checker_log.path.read_text().splitlines()]
    assert "checker_review_record_conflict" in events
    assert events.index("checker_error_record_committed") < events.index("checker_review_submitted")
