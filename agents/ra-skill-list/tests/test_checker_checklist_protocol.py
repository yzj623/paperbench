import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from research_agent.checker import (
    CheckAgentContext,
    CheckReport,
    CheckerResponseError,
    parse_check_report,
    parse_checklist_reviews,
)
from research_agent.plan import PlanTask
from research_agent.runtime_checker import RuntimeChecker
from research_agent.state_machine import ResearchAgentTools
from research_agent.storage import PlanStore


def _review(check_id, status="correct"):
    return {"id": check_id, "status": status, "reason": "paper/code comparison", "evidence": "paper.md:1 and submission/a.py:1"}


def _task(check_ids):
    return PlanTask.model_validate({
        "id": "task_001", "name": "test", "description": "test scope",
        "checklist": [
            {"id": check_id, "criterion": "requirement", "sources": ["paper.md:1"]}
            for check_id in check_ids
        ],
    })


def _response(*calls):
    return SimpleNamespace(
        tool_calls=[SimpleNamespace(name=name, arguments=args, call_id=f"call-{index}")
                    for index, (name, args) in enumerate(calls)],
        response_messages=[{"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call-{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for index, (name, args) in enumerate(calls)
        ]}],
    )


class _Config:
    def new_config(self, *, basicagent_tools, response_format):
        assert response_format is None
        assert [tool.name() for tool in basicagent_tools] == [
            "submit_checklist_reviews", "submit_unlisted_issues"
        ]
        return self

    def build(self):
        return self


def _checker(check_ids):
    task = _task(check_ids)
    context = CheckAgentContext(
        task=task, plan=[task.model_dump()], statuses={task.id: "uncompleted"},
        reproduction_description="submission/a.py", read_only_tools=[],
        read_only_dispatcher=SimpleNamespace(
            schemas={}, workspace=SimpleNamespace(workspace_root="/home")
        ),
    )
    return RuntimeChecker(completer_config=_Config(), context=context, max_steps=10), context


def test_checklist_ids_must_be_complete_unique_and_in_scope():
    ids = ["task_001_check_001", "task_001_check_002"]
    for reviews, expected in [
        ([_review(ids[0])], "missing"),
        ([_review(ids[0]), _review(ids[0]), _review(ids[1])], "duplicates"),
        ([_review(ids[0]), _review(ids[1]), _review("task_001_check_003")], "extra"),
    ]:
        with pytest.raises(CheckerResponseError, match=expected):
            parse_checklist_reviews({"reviews": reviews}, expected_ids=ids)
    assert len(parse_checklist_reviews({"reviews": [_review(ids[1]), _review(ids[0])]}, expected_ids=ids).reviews) == 2
    assert parse_checklist_reviews({"reviews": []}, expected_ids=[]).reviews == []


def test_program_decides_acceptance_and_returns_every_rejection():
    ids = [f"task_001_check_{n:03}" for n in range(1, 6)]
    reviews = [_review(ids[0], "correct"), _review(ids[1], "equivalent"),
               _review(ids[2], "checklist_error"), _review(ids[3], "approximate"),
               _review(ids[4], "incorrect")]
    report = parse_check_report({
        "reviews": reviews,
        "unlisted_issues": [{"description": "uncovered defect", "evidence": "submission/a.py:2"}],
    }, expected_ids=ids)
    assert not report.accepted
    assert len(report.issues) == 3
    assert ids[3] in report.issues[0].description
    assert ids[4] in report.issues[1].description
    assert report.issues[2].description == "uncovered defect"
    assert CheckReport(reviews=reviews[:3], unlisted_issues=[]).accepted
    assert CheckReport(reviews=[], unlisted_issues=[]).accepted


def test_checker_requires_order_and_skips_later_calls_in_same_response(monkeypatch):
    check_id = "task_001_check_001"
    responses = iter([
        _response(("submit_unlisted_issues", {"issues": []})),
        _response(("submit_checklist_reviews", {"reviews": []})),
        _response(("submit_checklist_reviews", {"reviews": [_review(check_id)]}),
                  ("submit_unlisted_issues", {"issues": []})),
        _response(("submit_unlisted_issues", {"issues": []})),
    ])

    async def fake_request(*, completer, messages):
        if len(messages) > 2 and messages[-1]["role"] == "tool":
            if messages[-1]["tool_call_id"] == "call-1":
                assert "MULTIPLE_TOOLS" in messages[-1]["content"]
        return next(responses)

    monkeypatch.setattr("research_agent.runtime_checker.make_completer_request", fake_request)
    checker, context = _checker([check_id])
    report = parse_check_report(asyncio.run(checker.review(context)), expected_ids=[check_id])
    assert report.accepted
    assert len(report.reviews) == 1


def test_empty_checklist_still_requires_both_tools(monkeypatch):
    responses = iter([
        _response(("submit_checklist_reviews", {"reviews": []})),
        _response(("submit_unlisted_issues", {"issues": [{"description": "missing", "evidence": "paper.md:1"}]})),
    ])

    async def fake_request(*, completer, messages):
        return next(responses)

    monkeypatch.setattr("research_agent.runtime_checker.make_completer_request", fake_request)
    checker, context = _checker([])
    report = parse_check_report(asyncio.run(checker.review(context)), expected_ids=[])
    assert not report.accepted
    assert report.issues[0].description == "missing"


def test_submit_subtask_forwards_all_program_rejection_reasons(tmp_path, monkeypatch):
    ids = ["task_001_check_001", "task_001_check_002", "task_001_check_003"]
    store = PlanStore(tmp_path / "run")
    store.initialize({"subtasks": [_task(ids).model_dump()]})
    report = CheckReport(
        reviews=[_review(ids[0], "checklist_error"), _review(ids[1], "approximate"),
                 _review(ids[2], "incorrect")],
        unlisted_issues=[{"description": "uncovered code defect", "evidence": "submission/a.py:9"}],
    )
    monkeypatch.setattr("research_agent.state_machine.review_with_retries", AsyncMock(return_value=report))
    manager = ResearchAgentTools(
        store=store, runtime=SimpleNamespace(active_processes=AsyncMock(return_value=[])),
        checker_factory=None, completer_config=None,
        public_paper_root=tmp_path, workspace_root=tmp_path,
    )
    feedback = asyncio.run(manager.submit_subtask(subtask_id="task_001", reproduction_description="a.py"))
    assert "uncompleted" in feedback
    assert ids[0] not in feedback
    assert ids[1] in feedback and ids[2] in feedback
    assert "uncovered code defect" in feedback
    assert feedback.count("Evidence:") == 3
