from copy import deepcopy

import pytest

from research_agent.plan import PlanValidationError, parse_plan
from research_agent.storage import PlanStore


def _plan():
    return {
        "subtasks": [
            {
                "id": "task_001",
                "name": "first",
                "description": "first scope",
                "checklist": [
                    {
                        "id": "task_001_check_001",
                        "criterion": "Use the method prescribed for this step.",
                        "sources": ["paper.md, section 2: short excerpt"],
                    }
                ],
            },
            {"id": "task_002", "name": "second", "description": "second scope", "checklist": []},
        ]
    }


def test_checklist_round_trip_and_render(tmp_path):
    plan = parse_plan(_plan())
    assert plan.model_dump() == _plan()
    store = PlanStore(tmp_path / "run")
    store.initialize(plan)
    assert store.load_plan().model_dump() == _plan()
    assert store.render()[0]["checklist"] == _plan()["subtasks"][0]["checklist"]


@pytest.mark.parametrize(
    "change",
    [
        lambda plan: plan["subtasks"][0].pop("checklist"),
        lambda plan: plan["subtasks"][0]["checklist"][0].update(id="task_002_check_001"),
        lambda plan: plan["subtasks"][0]["checklist"][0].update(id="check_001"),
        lambda plan: plan["subtasks"][0]["checklist"].append(
            deepcopy(plan["subtasks"][0]["checklist"][0])
        ),
        lambda plan: plan["subtasks"][0]["checklist"][0].update(criterion="  "),
        lambda plan: plan["subtasks"][0]["checklist"][0].update(sources=[]),
        lambda plan: plan["subtasks"][0]["checklist"][0].update(sources=[" "]),
        lambda plan: plan["subtasks"][0]["checklist"][0].update(extra="unexpected"),
    ],
)
def test_invalid_checklists_are_rejected(change):
    data = _plan()
    change(data)
    with pytest.raises(PlanValidationError):
        parse_plan(data)
