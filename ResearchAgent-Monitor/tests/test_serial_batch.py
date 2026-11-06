import importlib.util
import json
from pathlib import Path


def test_follow_batch_through_grading_next_run_and_completion(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "serial_batch_monitor", Path(__file__).resolve().parents[1] / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batch = tmp_path / "serial-batches/batch"
    first, second = batch / "01-planner-no-checker", batch / "02-planner-checker"
    for run in (first, second):
        state = run / "agent-run/research_agent"
        state.mkdir(parents=True)
        (state / "plan.json").write_text(json.dumps({"subtasks": [
            {"id": "task_001", "name": "task", "description": "implement"}]}))
        (run / "workspace/submission").mkdir(parents=True)
    summary = batch / "summary.json"
    records = [{"run_dir": str(first), "config": "planner-no-checker", "status": "generating"}]

    def save():
        summary.write_text(json.dumps({"paper_id": "all-in-one", "experiments": records}))

    save()
    monitor = module.Monitor(tmp_path, batch_summary=summary)
    monkeypatch.setattr(monitor, "_active_from_processes", lambda: (first, 123, 100))
    monkeypatch.setattr(module, "find_grading_process", lambda _: (None, None))
    value = monitor.refresh()
    assert value["run"]["paper_id"] == "all-in-one"
    assert value["run"]["config"] == "planner-no-checker"
    assert value["run"]["running"]

    monkeypatch.setattr(monitor, "_active_from_processes", lambda: (None, None, None))
    records[0]["status"] = "grading"
    save()
    value = monitor.refresh()
    assert value["run"]["run_dir"] == str(first)
    assert value["run"]["phase"] == "评分中"

    records[0]["status"] = "failed"
    records.append({"run_dir": str(second), "config": "planner-checker", "status": "generating"})
    save()
    value = monitor.refresh()
    assert value["run"]["run_dir"] == str(second)
    assert value["run"]["config"] == "planner-checker"

    grade = second / "evaluation/runs/group/paper/grade.json"
    grade.parent.mkdir(parents=True)
    grade.write_text(json.dumps({"paperbench_result": {"judge_output": {"score": 0.9}}}))
    records[-1]["status"] = "complete"
    save()
    value = monitor.refresh()
    assert value["run"]["run_dir"] == str(second)
    assert value["run"]["phase"] == "评分完成"
    assert value["evaluation"]["score_percent"] == 90.0
