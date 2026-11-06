import json

from research_agent.prompts import EXECUTION_PROMPT, render_plan_message
from research_agent.storage import PlanStore


def test_external_plan_initializes_store_and_execution_window(tmp_path):
    plan = {
        "subtasks": [
            {"id": "task_001", "name": "setup", "description": "foundation work"},
            {"id": "task_002", "name": "model", "description": "model implementation"},
        ]
    }
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan), encoding="utf-8")

    # Mirrors the solver's external-plan branch: read the file, then initialize
    # the store through the same strict-validating parse_plan path.
    store = PlanStore(tmp_path / "state")
    store.initialize(json.loads(plan_file.read_text(encoding="utf-8")))

    assert store.plan_path.exists()
    assert store.state_path.exists()
    assert store.first_uncompleted_task().id == "task_001"

    # Execution window content is built exactly as the resume/external branches
    # do: system = execution prompt, first user message = the frozen plan.
    message = render_plan_message(store.render())
    assert message["role"] == "user"
    assert "task_001" in message["content"]
    assert EXECUTION_PROMPT  # system prompt constant is imported and non-empty


def test_external_plan_strict_validation(tmp_path):
    from research_agent.plan import PlanValidationError

    bad = {"subtasks": [{"id": "not_a_task_id", "name": "x", "description": "y"}]}
    plan_file = tmp_path / "bad_plan.json"
    plan_file.write_text(json.dumps(bad), encoding="utf-8")

    store = PlanStore(tmp_path / "state")
    try:
        store.initialize(json.loads(plan_file.read_text(encoding="utf-8")))
    except PlanValidationError:
        pass
    else:
        raise AssertionError("expected strict plan validation to reject invalid plan")
