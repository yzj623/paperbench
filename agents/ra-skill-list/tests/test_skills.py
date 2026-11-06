from __future__ import annotations

import asyncio
from types import SimpleNamespace

from research_agent.context import MessageWindow
from research_agent.skills import (
    MAX_SELECTED_SKILLS,
    SelectSkillsTool,
    SkillCatalog,
    SkillCoordinator,
    SkillSelector,
)
from research_agent.storage import PlanStore


def test_disabled_skills_leave_only_checklist_prompt_and_base_tools():
    from research_agent.prompts import execution_prompt_for_skills
    from research_agent.solver import ResearchAgentSolver

    solver = ResearchAgentSolver(enable_skills=False)
    assert solver.shortname() == "researchagent"
    assert "ACTIVE_SKILLS" not in execution_prompt_for_skills(enabled=False)
    assert "current subtask's checklist" in execution_prompt_for_skills(enabled=False)
    assert [tool.name() for tool in solver._default_agent_tools()] == [
        tool.name() for tool in ResearchAgentSolver(enable_skills=True)._default_agent_tools()
    ]


def _catalog(tmp_path):
    root = tmp_path / "skills"
    first = root / "paper-a" / "method" / "first" / "SKILL.md"
    second = root / "paper-a" / "method" / "second" / "SKILL.md"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("---\nname: First skill\ndescription: first description\n---\n# First\nBody A")
    second.write_text("# Second skill\n\nSecond description.")
    (second.parent / "script.py").write_text("must not be loaded")
    return SkillCatalog(root=root, target="paper-a")


def test_selector_sees_all_tasks_without_changing_main_visibility(tmp_path):
    from research_agent.prompts import render_plan_message

    store = PlanStore(tmp_path / "run")
    store.initialize({"subtasks": [
        {"id": "task_001", "name": "first", "description": "first scope", "checklist": []},
        {"id": "task_002", "name": "later", "description": "unique later scope", "checklist": []},
    ]})
    selector = SkillSelector(
        catalog=_catalog(tmp_path), config=None, read_only_tools=[],
        computer=None, store=store, workspace_path="/home",
    )
    prompt = selector._system_prompt()
    assert "task_001" in prompt
    assert "task_002" in prompt
    assert "unique later scope" in prompt
    assert "later subtasks hidden" not in prompt
    main_plan = render_plan_message(store.render())["content"]
    assert "task_001" in main_plan
    assert "task_002" not in main_plan


def test_catalog_loads_only_skill_text_and_stable_ids(tmp_path):
    catalog = _catalog(tmp_path)
    assert set(catalog.documents) == {"method/first", "method/second"}
    assert catalog.documents["method/first"].name == "First skill"
    assert "must not be loaded" not in "\n".join(
        item.content for item in catalog.documents.values()
    )
    assert all(item["skill_id"] for item in catalog.metadata())


def test_selection_tool_returns_errors_and_accepts_zero_to_four(tmp_path):
    catalog = _catalog(tmp_path)
    tool = SelectSkillsTool(catalog=catalog, task_ids=["task_001", "task_002"])
    invalid = asyncio.run(
        tool.execute(
            None,
            task_selections=[
                {
                    "task_id": "task_001",
                    "skills": [{"skill_id": "missing", "reason": "looks useful"}],
                }
            ],
        )
    )
    assert "Unknown skill ID" in invalid
    assert tool.selected is None
    assert tool.selection_attempts == 1
    selected = [
        {
            "task_id": "task_001",
            "skills": [{"skill_id": "method/first", "reason": "directly applicable"}],
        },
        {
            "task_id": "task_002",
            "skills": [{"skill_id": "method/second", "reason": "validation help"}],
        },
    ]
    valid = asyncio.run(tool.execute(None, task_selections=selected))
    assert '"ok": true' in valid
    assert set(tool.selected) == {"task_001", "task_002"}
    assert all(len(items) <= MAX_SELECTED_SKILLS for items in tool.selected.values())


def test_skill_slot_survives_trimming_and_is_replaced():
    window = MessageWindow(
        {"role": "system", "content": "system"},
        {"role": "user", "content": "plan"},
    )
    window.update_skill("skill one")
    window.extend(
        [
            {"role": "assistant", "content": "old"},
            {"role": "user", "content": "new"},
        ]
    )
    window.trim_for_context(0)
    assert [message["content"] for message in window.messages] == [
        "system",
        "plan",
        "skill one",
    ]
    window.update_skill("skill two")
    assert window.messages[2]["content"] == "skill two"


def test_coordinator_selects_all_tasks_once_and_restores(tmp_path):
    catalog = _catalog(tmp_path)
    store = PlanStore(tmp_path / "run")
    store.initialize(
        {
            "subtasks": [
                {"id": "task_001", "name": "one", "description": "first", "checklist": []},
                {"id": "task_002", "name": "two", "description": "second", "checklist": []},
            ]
        }
    )

    class FakeSelector:
        def __init__(self):
            self.calls = []

        async def select(self):
            self.calls.append("all")
            return {
                "task_001": [{"skill_id": "method/first", "reason": "relevant"}],
                "task_002": [{"skill_id": "method/second", "reason": "relevant"}],
            }

    selector = FakeSelector()
    coordinator = SkillCoordinator(catalog=catalog, selector=selector, store=store)
    session = SimpleNamespace(
        window=MessageWindow(
            {"role": "system", "content": "system"},
            {"role": "user", "content": "plan"},
        )
    )
    asyncio.run(coordinator.ensure_current(session))
    asyncio.run(coordinator.ensure_current(session))
    assert selector.calls == ["all"]
    assert "Body A" in session.window.messages[2]["content"]

    restored_selector = FakeSelector()
    restored = SkillCoordinator(catalog=catalog, selector=restored_selector, store=store)
    asyncio.run(restored.ensure_current(session))
    assert restored_selector.calls == []

    store.set_status("task_001", "completed")
    asyncio.run(coordinator.ensure_current(session))
    assert selector.calls == ["all"]
    assert "Second description" in session.window.messages[2]["content"]

