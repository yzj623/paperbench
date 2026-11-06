"""Verify both checker material modes without API calls or an experiment."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from research_agent.checker import CheckAgentContext
from research_agent.context import MessageWindow
from research_agent.plan import PlanTask
from research_agent.prompts import render_checker_prompt
from research_agent.readonly import PathAccessError, ReadOnlyToolDispatcher, SafeWorkspace
from research_agent.runtime_checker import RuntimeChecker


class _Config:
    context_window = 256_000

    def new_config(self, *, basicagent_tools, response_format):
        assert response_format is None
        return self

    def build(self):
        return self


def _response(name: str, arguments: dict):
    return SimpleNamespace(
        tool_calls=[SimpleNamespace(name=name, arguments=arguments, call_id="call-1")],
        response_messages=[{"role": "assistant", "content": "", "tool_calls": [{
            "id": "call-1", "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            },
        }]}],
    )


def _workspace(tmp_path, *, embedded: bool):
    paper = tmp_path / "paper"
    paper.mkdir(exist_ok=True)
    (paper / "paper.md").write_text("Distinct paper source.\n", encoding="utf-8")
    (paper / "addendum.md").write_text("Distinct addendum source.\n", encoding="utf-8")
    submission = tmp_path / "submission"
    submission.mkdir(exist_ok=True)
    (submission / "model.py").write_text("answer = 42\n", encoding="utf-8")
    briefing = tmp_path / "check_instruction.txt"
    briefing.write_text("Distinct checker briefing.\n", encoding="utf-8")
    return SafeWorkspace(
        public_root=paper, workspace_root=tmp_path,
        submission_root=submission, checker_instructions_file=briefing,
        checker_submission_only=embedded,
    )


def _context(workspace, task_id):
    task = PlanTask(id=task_id, name="Inspect", description="Inspect code", checklist=[])
    return CheckAgentContext(
        task=task, plan=[task.model_dump()], statuses={task_id: "uncompleted"},
        reproduction_description="submission/model.py", read_only_tools=[],
        read_only_dispatcher=ReadOnlyToolDispatcher(workspace),
    )


def test_embedded_mode_keeps_cross_task_prefix_and_task_after_trim(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path, embedded=True)
    captured = []
    calls = iter([_response("submit_checklist_reviews", {"reviews": []}),
                  _response("submit_unlisted_issues", {"issues": []})] * 2)

    async def fake_request(*, completer, messages):
        captured.append(deepcopy(messages))
        return next(calls)

    monkeypatch.setattr(
        "research_agent.runtime_checker.make_completer_request", fake_request
    )
    for task_id in ("task_001", "task_002"):
        context = _context(workspace, task_id)
        report = asyncio.run(RuntimeChecker(
            completer_config=_Config(), context=context,
            materials_mode="embedded",
        ).review(context))
        assert report == {"reviews": [], "unlisted_issues": []}
    first, second = captured[0], captured[2]
    assert first[:2] == second[:2]
    assert first[2] != second[2]
    assert all(name in first[1]["content"] for name in (
        "Distinct paper source", "Distinct addendum source", "Distinct checker briefing"
    ))
    assert "task_001" in first[2]["content"]
    assert "task_002" in second[2]["content"]
    assert "Do not call list_directory" in first[0]["content"]
    window = MessageWindow(first[0], first[1], protected_context_message=first[2])
    window.append({"role": "user", "content": "old review history"})
    assert window.trim_oldest_unit()
    assert window.messages == first[:3]


def test_embedded_tools_reject_fixed_files_but_allow_submission(tmp_path):
    embedded = ReadOnlyToolDispatcher(_workspace(tmp_path, embedded=True))
    assert "answer = 42" in embedded.call(
        "read_file", {"path": "/home/submission/model.py"}
    )
    for path in ("/home/paper", "/home/paper/paper.md", "/home/paper/addendum.md",
                 "/check_instruction.txt"):
        with pytest.raises(PathAccessError):
            embedded.call("read_file" if path != "/home/paper" else "list_directory",
                          {"path": path})
    with pytest.raises(PathAccessError):
        embedded.call("search_file", {"path": "/home/paper/paper.md", "query": "source"})
    assert "model.py" in embedded.call(
        "list_directory", {"path": "/home/submission"}
    )


def test_tools_mode_retains_original_paper_access_and_prompt(tmp_path):
    ordinary = ReadOnlyToolDispatcher(_workspace(tmp_path, embedded=False))
    assert "Distinct paper source" in ordinary.call(
        "read_file", {"path": "/home/paper/paper.md"}
    )
    assert "Distinct checker briefing" in ordinary.call(
        "read_file", {"path": "/check_instruction.txt"}
    )
    prompt = render_checker_prompt("/home")
    assert "Read the public paper and addendum" in prompt
    assert "__CHECKER_MATERIALS_ACCESS__" not in prompt

