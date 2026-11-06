import json
from pathlib import Path
import subprocess
import os

import pytest

from research_agent.resume import locate_source, prepare_resume, resume_message
from research_agent.storage import PlanStore, atomic_json_write


def source_run(tmp_path):
    root = tmp_path / "example-paper" / "old-run"
    submission = root / "workspace/submission"
    submission.mkdir(parents=True)
    (submission / "model.py").write_text("answer = 42\n")
    plan = {"subtasks": [
        {"id": "task_001", "name": "first", "description": "first", "checklist": []},
        {"id": "task_002", "name": "second", "description": "second", "checklist": []},
    ]}
    store = PlanStore(root / "agent-run/research_agent")
    store.initialize(plan)
    store.set_status("task_001", "completed")
    store.record_failure("task_002", increment=True)
    atomic_json_write(store.skill_selection_path, {"task_selections": {"task_001": [], "task_002": []}})
    store.tool_log.append("tool_result", tool="submit_subtask", result={
        "subtask_id": "task_002", "accepted": False, "message": "Fix the missing branch."
    })
    atomic_json_write(store.jobs_path, {"jobs": [{"pid": 123, "status": "running"}]})
    (root / "agent-run/rollout-error.json").write_text('{}')
    return root, store


@pytest.mark.parametrize("kind", ["run", "workspace", "submission"])
def test_resume_copies_files_and_state_without_old_processes_or_history(tmp_path, kind):
    old, previous = source_run(tmp_path)
    source = {"run": old, "workspace": old / "workspace",
              "submission": old / "workspace/submission"}[kind]
    before = previous.state_path.read_bytes()
    run = prepare_resume(paper_id="example-paper", source=source,
                         experiments_root=tmp_path / "new")
    store = PlanStore(run / "agent-run/research_agent")
    assert store.status("task_001") == "completed"
    assert store.first_uncompleted_task().id == "task_002"
    assert store.failure_count("task_002") == 1
    assert (run / "workspace/submission/model.py").read_text() == "answer = 42\n"
    assert store.skill_selection_path.read_bytes() == previous.skill_selection_path.read_bytes()
    assert not store.jobs_path.exists()
    assert not store.tool_log.path.exists()
    assert not store.checker_log.path.exists()
    assert not (run / "agent-run/rollout-error.json").exists()
    message = resume_message(store.run_dir, "/home")
    assert message["role"] == "user" and "Fix the missing branch." in message["content"]
    assert "new execution session" in message["content"]
    (run / "workspace/submission/model.py").write_text("new = 99\n")
    assert (old / "workspace/submission/model.py").read_text() == "answer = 42\n"
    assert previous.state_path.read_bytes() == before


def test_bare_submission_requires_plan_and_can_start_without_approvals(tmp_path):
    old, store = source_run(tmp_path)
    code = tmp_path / "bare"
    code.mkdir()
    (code / "code.py").write_text("pass\n")
    with pytest.raises(ValueError, match="provide --plan"):
        prepare_resume(paper_id="example-paper", source=code, experiments_root=tmp_path / "new")
    assert not (tmp_path / "new").exists()
    run = prepare_resume(paper_id="example-paper", source=code,
                         experiments_root=tmp_path / "new", plan_path=store.plan_path)
    restored = PlanStore(run / "agent-run/research_agent")
    assert restored.first_uncompleted_task().id == "task_001"
    assert not restored.submission_failures()


def test_changed_plan_does_not_inherit_approvals(tmp_path):
    old, store = source_run(tmp_path)
    plan = store.load_plan().model_dump()
    plan["subtasks"][0]["description"] = "different requirements"
    override = tmp_path / "plan.json"
    atomic_json_write(override, plan)
    run = prepare_resume(paper_id="example-paper", source=old,
                         experiments_root=tmp_path / "new", plan_path=override)
    restored = PlanStore(run / "agent-run/research_agent")
    assert all(s == "uncompleted" for s in restored.statuses().values())
    assert not restored.skill_selection_path.exists()


def test_explicit_state_and_invalid_state(tmp_path):
    old, store = source_run(tmp_path)
    code = tmp_path / "bare"
    code.mkdir()
    run = prepare_resume(paper_id="example-paper", source=code,
                         experiments_root=tmp_path / "new", plan_path=store.plan_path,
                         state_path=store.state_path)
    assert PlanStore(run / "agent-run/research_agent").status("task_001") == "completed"
    bad = tmp_path / "bad.json"
    atomic_json_write(bad, {"statuses": {"task_999": "completed"}})
    with pytest.raises(ValueError, match="match the plan"):
        prepare_resume(paper_id="example-paper", source=old,
                       experiments_root=tmp_path / "invalid", state_path=bad)
    assert not (tmp_path / "invalid").exists()
    with pytest.raises(ValueError, match="different paper"):
        prepare_resume(paper_id="another-paper", source=old, experiments_root=tmp_path / "invalid")


def test_resume_entrypoint_passes_seeded_run_to_agent_then_grader(tmp_path):
    from test_experiment_script import _fake_root

    old, store = source_run(tmp_path)
    fake = _fake_root(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/resume_experiment.sh"
    env = {**os.environ, "RESEARCH_AGENT_ROOT": str(fake),
           "EXPERIMENTS_ROOT": str(tmp_path / "outputs"),
           "PAPERBENCH_GRADER_API_CONFIG": "openai", "GRADER_OPENAI_API_KEY": "test-only"}
    command = ["bash", str(script), "example-paper", str(old)]
    first = subprocess.run(command, env=env, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    second = subprocess.run(command + ["--prepare-only"], env=env, text=True, capture_output=True)
    assert second.returncode == 0, second.stderr
    runs = list((tmp_path / "outputs/example-paper").iterdir())
    assert len(runs) == 2
    graded = [r for r in runs if (r / "evaluation/test-result").exists()]
    assert len(graded) == 1
    assert (graded[0] / "workspace/submission/model.py").exists()
    assert PlanStore(graded[0] / "agent-run/research_agent").first_uncompleted_task().id == "task_002"
