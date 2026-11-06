"""Offline scoring contracts shared by the four independent monitor copies."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[2]
MONITORS = ["ResearchAgent-Monitor", "ResearchAgent-Skill-List-Monitor", "Pi-Monitor", "codex-agent-prototype/monitor"]


@pytest.fixture(params=MONITORS)
def server(request):
    path = LAB / request.param / "server.py"
    spec = importlib.util.spec_from_file_location("scoring_" + request.param.replace("/", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def report(run, name="group", base="evaluation"):
    return run / base / "runs" / name / "paper-try" / "grade.json"


def test_missing_and_partial_reports_are_not_zero(server, tmp_path):
    assert server.read_evaluation(tmp_path)["score_percent"] is None
    path = report(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"score":')
    value = server.read_evaluation(tmp_path, grading_active=True)
    assert value["score_percent"] is None
    assert value["status"] == "grading"
    assert len(value["invalid_reports"]) == 1


def test_both_grade_formats_and_valid_zero(server, tmp_path):
    path = report(tmp_path)
    save(path, {"score": 0})
    assert server.read_evaluation(tmp_path)["score_percent"] == 0
    save(path, {"paperbench_result": {"judge_output": {"score": 0.9166666666666666,
                                                         "num_invalid_leaf_nodes": 2}}})
    value = server.read_evaluation(tmp_path)
    assert value["score_percent"] == 91.67
    assert value["invalid_leaf_nodes"] == 2
    assert value["status"] == "complete"


def test_pi_grade_directory_and_latest_by_time(server, tmp_path):
    first = report(tmp_path, "z-last-name", base="grade")
    second = report(tmp_path, "a-first-name", base="grade")
    save(first, {"score": 0.7})
    save(second, {"score": 0.9})
    os.utime(first, (100, 100))
    os.utime(second, (200, 200))
    value = server.read_evaluation(tmp_path)
    assert value["score_percent"] == 90
    assert len(value["reports"]) == 2
    assert value["latest_report"]["path"].startswith("grade/")


@pytest.mark.parametrize("value", [{"score": float("nan")}, {"score": True}, {"score": 1.5},
                                   {"score": 0, "paperbench_result": {"grading_failed": True}}])
def test_bad_score_is_not_a_completed_grade(server, tmp_path, value):
    save(report(tmp_path), value)
    result = server.read_evaluation(tmp_path)
    assert result["score_percent"] is None
    assert result["status"] != "complete"


def test_failed_grading_is_explicit(server, tmp_path):
    result = server.read_evaluation(tmp_path, host_phase="grading_failed")
    assert result["score_percent"] is None
    assert result["status"] == "failed"


def test_report_symlink_cannot_escape_run(server, tmp_path):
    external = tmp_path / "outside.json"
    save(external, {"score": 1})
    run = tmp_path / "run"
    path = report(run)
    path.parent.mkdir(parents=True)
    path.symlink_to(external)
    assert server.read_evaluation(run)["reports"] == []


def test_ra_stays_on_last_run_and_can_pin(tmp_path, monkeypatch):
    path = LAB / "ResearchAgent-Monitor/server.py"
    spec = importlib.util.spec_from_file_location("ra_selection_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task = {"id": "task_001", "name": "task", "description": "implement"}
    old, current = tmp_path / "all-in-one/old", tmp_path / "mechanistic-understanding/current"
    for run in [old, current]:
        save(run / "agent-run/research_agent/plan.json", {"subtasks": [task]})
        save(run / "agent-run/research_agent/plan_state.json", {"statuses": {task["id"]: "uncompleted"}})
        (run / "workspace/submission").mkdir(parents=True)
    monkeypatch.setattr(module, "find_grading_process", lambda _: (None, None))
    monitor = module.Monitor(tmp_path)
    monkeypatch.setattr(monitor, "_active_from_processes", lambda: (current, 123, 100))
    assert monitor.refresh()["run"]["run_dir"] == str(current)
    monkeypatch.setattr(monitor, "_active_from_processes", lambda: (None, None, None))
    save(report(current), {"score": 0.9166666666666666})
    value = monitor.refresh()
    assert value["run"]["run_dir"] == str(current)
    assert value["run"]["phase"] == "评分完成"
    assert value["evaluation"]["score_percent"] == 91.67
    pinned = module.Monitor(tmp_path, current)
    monkeypatch.setattr(pinned, "_active_from_processes", lambda: (None, None, None))
    assert pinned.refresh()["run"]["run_dir"] == str(current)
