import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from research_agent.prompts import render_plan_message
from research_agent.session import ResearchAgentSession
from research_agent.state_machine import ResearchAgentTools, ResearchToolError
from research_agent.storage import PlanStore


@pytest.fixture
def store(tmp_path):
    store = PlanStore(tmp_path / "state")
    store.initialize({"subtasks": [
        {"id": f"task_{i:03}", "name": f"unique_name_{i}", "description": f"unique_scope_{i}"}
        for i in range(1, 4)
    ]})
    return store


def test_prefix_and_restored_state(store):
    original = store.plan_path.read_bytes()
    text = render_plan_message(store.render())["content"]
    assert "unique_scope_1" in text
    assert "unique_scope_2" not in text
    assert "2 later subtasks hidden" in text
    store.set_status("task_001", "completed")
    store.set_status("task_002", "skipped")
    restored = PlanStore(store.run_dir)
    text = render_plan_message(restored.render())["content"]
    assert all(f"unique_scope_{i}" in text for i in range(1, 4))
    assert "no later hidden subtasks" in text
    store.set_status("task_003", "completed")
    assert restored.first_uncompleted_task() is None
    assert store.plan_path.read_bytes() == original


def test_planning_history_is_discarded(tmp_path):
    store = PlanStore(tmp_path / "new")
    plan = {"subtasks": [
        {"id": f"task_{i:03}", "name": f"name_{i}", "description": f"scope_{i}"}
        for i in range(1, 3)
    ]}
    session = ResearchAgentSession(
        store=store, completer=SimpleNamespace(complete=AsyncMock(return_value=plan))
    )
    asyncio.run(session.create_plan(public_material_message="public materials"))
    assert "scope_2" not in session.execution_messages()[1]["content"]
    store.set_status("task_001", "completed")
    assert "scope_2" in session.execution_messages()[1]["content"]


def test_protocol_advances_on_direct_submission_or_valid_skip(store, tmp_path):
    manager = ResearchAgentTools(
        store=store, runtime=SimpleNamespace(active_processes=AsyncMock(return_value=[])),
        public_paper_root=tmp_path, workspace_root=tmp_path,
    )

    async def exercise():
        for action in (
            manager.submit_subtask(subtask_id="task_002", reproduction_description="evidence"),
            manager.skip_subtask(subtask_id="task_002", reason="blocked"),
        ):
            with pytest.raises(ResearchToolError, match="Only the current"):
                await action
        await manager.skip_subtask(subtask_id="task_001", reason="blocked")
        assert store.first_uncompleted_task().id == "task_001"
        await manager.submit_subtask(subtask_id="task_001", reproduction_description="evidence")
        assert store.first_uncompleted_task().id == "task_002"
        assert store.status("task_001") == "completed"
        assert not store.checker_log.path.exists()
        for _ in range(3):
            store.record_failure("task_002", increment=True)
        await manager.skip_subtask(subtask_id="task_002", reason="unavailable public input")
        assert store.first_uncompleted_task().id == "task_003"
        with pytest.raises(ResearchToolError, match="Only the current"):
            await manager.submit_subtask(subtask_id="task_002", reproduction_description="retry")
        assert not (await manager.final_submit()).accepted

    asyncio.run(exercise())
