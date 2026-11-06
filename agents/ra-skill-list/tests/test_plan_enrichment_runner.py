import asyncio
import json
from pathlib import Path

import pytest

from research_agent.plan_enrichment_runner import run


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "paper").mkdir(parents=True)
    (root / "submission").mkdir()
    (root / "paper" / "paper.md").write_text("Public paper", encoding="utf-8")
    (root / "paper" / "addendum.md").write_text("Public addendum", encoding="utf-8")
    (root / "instructions.txt").write_text("Code-Dev", encoding="utf-8")
    return root


def _outline(tmp_path: Path) -> Path:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"subtasks": [{
        "id": "task_001", "name": "fixed name", "description": "fixed description"
    }]}), encoding="utf-8")
    return path


def test_real_runner_stops_after_valid_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-value")
    async def fake_agent(*, config, outline, workspace_root, store, max_steps):
        assert "Public paper" in (workspace_root / "paper" / "paper.md").read_text()
        assert [task.id for task in outline.subtasks] == ["task_001"]
        plan = {"subtasks": [{**outline.subtasks[0].model_dump(), "checklist": [{
            "id": "task_001_check_001", "criterion": "Paper-prescribed method is implemented.",
            "sources": [{"document": "paper.md", "quote": "Public paper"}],
        }]}]}
        return store.initialize(plan)

    monkeypatch.setattr("research_agent.plan_enrichment_runner.enrich_legacy_outline_agent", fake_agent)
    run_dir = tmp_path / "run"
    path = asyncio.run(run(
        workspace=_workspace(tmp_path), outline_path=_outline(tmp_path), run_dir=run_dir
    ))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["subtasks"][0]["name"] == "fixed name"
    assert saved["subtasks"][0]["checklist"][0]["id"] == "task_001_check_001"
    assert not (run_dir / "submissions").exists()
    assert not any((tmp_path / "workspace" / "submission").iterdir())


def test_runner_allows_empty_checklists(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-value")
    async def fake_agent(*, config, outline, workspace_root, store, max_steps):
        return store.initialize({"subtasks": [{**outline.subtasks[0].model_dump(), "checklist": []}]})

    monkeypatch.setattr("research_agent.plan_enrichment_runner.enrich_legacy_outline_agent", fake_agent)
    run_dir = tmp_path / "run"
    path = asyncio.run(run(
        workspace=_workspace(tmp_path), outline_path=_outline(tmp_path), run_dir=run_dir
    ))
    assert json.loads(path.read_text())["subtasks"][0]["checklist"] == []
