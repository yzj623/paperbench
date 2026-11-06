import json
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_experiment.sh"


def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "fake-root"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (root / "src").symlink_to(REPO / "src", target_is_directory=True)
    (root / ".venv-wsl").symlink_to(REPO / ".venv-wsl", target_is_directory=True)
    agent = scripts / "run_agent.sh"
    agent.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "[[ $# -eq 3 ]]\n[[ -f \"$3\" ]]\n"
        "mkdir -p \"$2/fake-submission\"\n"
        "printf '%s\\n' \"$2/fake-submission\" > \"$2/FINAL_SUBMISSION_PATH\"\n",
        encoding="utf-8",
    )
    agent.chmod(0o755)
    judge = scripts / "evaluate_submission.sh"
    judge.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "[[ $# -eq 3 ]]\n[[ -d \"$2\" ]]\n"
        "mkdir -p \"$3\"\nprintf 'scored\\n' > \"$3/test-result\"\n",
        encoding="utf-8",
    )
    judge.chmod(0o755)
    return root


def _run(tmp_path: Path, root: Path, outline: Path):
    env = os.environ.copy()
    env.update({
        "RESEARCH_AGENT_ROOT": str(root),
        "EXPERIMENTS_ROOT": str(tmp_path / "experiments"),
        "PAPERBENCH_GRADER_API_CONFIG": "openai",
        "GRADER_OPENAI_API_KEY": "test-only",
    })
    return subprocess.run(
        ["bash", str(SCRIPT), "example-paper", str(outline)],
        env=env, text=True, capture_output=True, check=False,
    )


def test_one_command_passes_fixed_outline_and_runs_grading_without_overwrite(tmp_path):
    root = _fake_root(tmp_path)
    outline = tmp_path / "legacy outline.json"
    data = {"subtasks": [{"id": "task_001", "name": "A", "description": "Implement A"}]}
    outline.write_text(json.dumps(data), encoding="utf-8")
    first = _run(tmp_path, root, outline)
    second = _run(tmp_path, root, outline)
    assert first.returncode == second.returncode == 0, (first.stderr, second.stderr)
    runs = list((tmp_path / "experiments" / "example-paper").iterdir())
    assert len(runs) == 2
    for run in runs:
        assert json.loads((run / "input_outline.json").read_text()) == data
        assert json.loads((run / "input_plan.json").read_text()) == data
        assert (run / "evaluation" / "test-result").read_text() == "scored\n"


def test_complete_checklist_plan_runs_without_enrichment_and_is_archived(tmp_path):
    root = _fake_root(tmp_path)
    plan = tmp_path / "complete.json"
    data = {"subtasks": [{"id": "task_001", "name": "A", "description": "A",
                         "checklist": [{"id": "task_001_check_001", "criterion": "Check A",
                                        "sources": [{"document": "paper.md", "quote": "Source A"}]}]}]}
    plan.write_text(json.dumps(data), encoding="utf-8")
    result = _run(tmp_path, root, plan)
    assert result.returncode == 0, result.stderr
    runs = list((tmp_path / "experiments" / "example-paper").iterdir())
    assert len(runs) == 1
    assert json.loads((runs[0] / "input_plan.json").read_text()) == data
    assert not (runs[0] / "input_outline.json").exists()
    assert (runs[0] / "evaluation" / "test-result").read_text() == "scored\n"


def test_mechanistic_diagnostic_plan_is_accepted_by_one_command_entrypoint(tmp_path):
    root = _fake_root(tmp_path)
    plan = REPO / "plans/diagnostic/mechanistic-rubric-assisted/plan.json"
    result = _run(tmp_path, root, plan)
    assert result.returncode == 0, result.stderr
    runs = list((tmp_path / "experiments" / "example-paper").iterdir())
    assert len(runs) == 1
    assert (runs[0] / "input_plan.json").read_bytes() == plan.read_bytes()
    assert not (runs[0] / "input_outline.json").exists()
    assert (runs[0] / "evaluation" / "test-result").read_text() == "scored\n"


def test_rejects_invalid_checklist_plan_before_creating_run(tmp_path):
    root = _fake_root(tmp_path)
    plan = tmp_path / "wrong.json"
    plan.write_text(json.dumps({"subtasks": [{
        "id": "task_001", "name": "A", "description": "A", "checklist": [
            {"id": "task_002_check_001", "criterion": "Check A", "sources": ["paper.md"]}
        ]
    }]}), encoding="utf-8")
    result = _run(tmp_path, root, plan)
    assert result.returncode == 2
    assert "Invalid plan" in result.stderr
    assert not (tmp_path / "experiments").exists()
