import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent.plan import PlanValidationError, parse_plan
from research_agent.plan_enrichment import (
    ChecklistTextCompleter,
    enrich_legacy_outline,
    is_legacy_outline,
    merge_checklists,
    parse_legacy_outline,
)
from research_agent.storage import PlanStore


def _legacy():
    return {"subtasks": [
        {"id": "task_001", "name": "first", "description": "original first scope"},
        {"id": "task_002", "name": "second", "description": "original second scope"},
    ]}


def _response():
    return {"checklists": [
        {"task_id": "task_001", "checklist": [{
            "id": "task_001_check_001",
            "criterion": "Use the paper-prescribed method for this step.",
            "sources": ["paper.md, section 2: short original excerpt"],
        }]},
        {"task_id": "task_002", "checklist": []},
    ]}


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "paper").mkdir(parents=True)
    (root / "paper" / "paper.md").write_text("FULL PAPER TEXT unique-paper-marker", encoding="utf-8")
    (root / "paper" / "addendum.md").write_text("FULL ADDENDUM unique-addendum-marker", encoding="utf-8")
    (root / "instructions.txt").write_text("FULL CODE-DEV unique-instruction-marker", encoding="utf-8")
    return root


def test_branch_selection_and_legacy_validation():
    assert is_legacy_outline(_legacy())
    complete = deepcopy(_legacy())
    for task in complete["subtasks"]:
        task["checklist"] = []
    assert not is_legacy_outline(complete)
    assert parse_plan(complete).subtasks[0].checklist == []
    del complete["subtasks"][1]["checklist"]
    assert not is_legacy_outline(complete)
    with pytest.raises(PlanValidationError):
        parse_plan(complete)
    invalid = _legacy()
    invalid["subtasks"][0]["unexpected"] = "value"
    with pytest.raises(PlanValidationError):
        parse_legacy_outline(invalid)


def test_merge_preserves_old_outline_and_round_trips(tmp_path):
    original = _legacy()
    plan = merge_checklists(json.dumps(_response()), parse_legacy_outline(original))
    assert [{"id": t.id, "name": t.name, "description": t.description} for t in plan.subtasks] == original["subtasks"]
    assert plan.subtasks[0].checklist[0].id == "task_001_check_001"
    store = PlanStore(tmp_path / "run")
    store.initialize(plan)
    assert store.load_plan() == plan
    assert original == _legacy()


@pytest.mark.parametrize("change", [
    lambda value: value["checklists"].reverse(),
    lambda value: value["checklists"].pop(),
    lambda value: value["checklists"][0].update(task_id="task_999"),
    lambda value: value["checklists"][0].update(name="rewritten"),
    lambda value: value["checklists"][0]["checklist"][0].update(id="task_002_check_001"),
    lambda value: value["checklists"][0]["checklist"][0].update(sources=[]),
])
def test_invalid_enrichment_is_rejected(change):
    response = _response()
    change(response)
    with pytest.raises(PlanValidationError):
        merge_checklists(response, parse_legacy_outline(_legacy()))


def test_enrichment_retries_and_returns_complete_plan(tmp_path):
    class FakeCompleter:
        def __init__(self):
            self.calls = 0

        async def complete(self, messages, *, tools):
            self.calls += 1
            assert tools == []
            assert '"original first scope"' in messages[1]["content"]
            for marker in ("unique-paper-marker", "unique-addendum-marker", "unique-instruction-marker"):
                assert marker in messages[1]["content"]
            assert "you have no file-reading tools" in messages[0]["content"]
            if self.calls == 1:
                return '{"checklists": []}'
            assert "INVALID_CHECKLISTS" in messages[-1]["content"]
            return json.dumps(_response())

    fake = FakeCompleter()
    plan = asyncio.run(enrich_legacy_outline(
        completer=fake,
        outline=parse_legacy_outline(_legacy()),
        workspace_root=_workspace(tmp_path),
        max_retries=1,
    ))
    assert fake.calls == 2
    assert [task.id for task in plan.subtasks] == ["task_001", "task_002"]


def test_enrichment_failure_does_not_create_plan(tmp_path):
    class InvalidCompleter:
        async def complete(self, messages, *, tools):
            return '{"checklists": []}'

    store = PlanStore(tmp_path / "run")
    with pytest.raises(PlanValidationError):
        asyncio.run(enrich_legacy_outline(
            completer=InvalidCompleter(),
            outline=parse_legacy_outline(_legacy()),
            workspace_root=_workspace(tmp_path),
            max_retries=0,
        ))
    assert not store.plan_path.exists()


def test_text_completer_makes_one_request_without_tools(monkeypatch):
    calls = []

    async def fake_request(*, completer, messages):
        calls.append((completer, messages))
        return SimpleNamespace(
            tool_calls=[],
            response_messages=[{"role": "assistant", "content": json.dumps(_response())}],
        )

    monkeypatch.setattr("paperbench.solvers.basicagent.api.make_completer_request", fake_request)
    client = object()
    config = SimpleNamespace(build=lambda: client)
    messages = [{"role": "system", "content": "instructions"}, {"role": "user", "content": "paper"}]
    result = asyncio.run(ChecklistTextCompleter(config).complete(messages, tools=[]))
    assert json.loads(result) == _response()
    assert calls == [(client, messages)]
