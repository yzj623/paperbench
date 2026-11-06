import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent.plan_enrichment import (
    CHECKLIST_ENRICHMENT_PROMPT,
    CHECKLIST_ENRICHMENT_REQUEST,
    parse_legacy_outline,
)
from research_agent.plan_enrichment_agent import (
    SubmitInput,
    _normalized_quote,
    _unmatched_sources,
    enrich_legacy_outline_agent,
)
from research_agent.storage import PlanStore


def _workspace(root: Path) -> Path:
    workspace = root / "workspace"
    (workspace / "paper").mkdir(parents=True)
    (workspace / "paper" / "paper.md").write_text("paper text", encoding="utf-8")
    (workspace / "paper" / "addendum.md").write_text("addendum text", encoding="utf-8")
    (workspace / "instructions.txt").write_text("instructions text", encoding="utf-8")
    return workspace


def _outline():
    return parse_legacy_outline({"subtasks": [
        {"id": "task_001", "name": "A", "description": "first"},
        {"id": "task_002", "name": "B", "description": "second"},
    ]})


def _call(name, arguments, number):
    return SimpleNamespace(name=name, arguments=arguments, call_id=f"call-{number}")


def _response(*calls):
    return SimpleNamespace(
        tool_calls=list(calls),
        response_messages=[{"role": "assistant", "content": "", "tool_calls": [
            {"id": call.call_id, "type": "function", "function": {
                "name": call.name, "arguments": json.dumps(call.arguments),
            }} for call in calls
        ]}],
    )


class _Config:
    def new_config(self, *, basicagent_tools):
        assert [tool.name() for tool in basicagent_tools] == ["submit_task_checklist", "finish_plan"]
        return self

    def build(self):
        return self


def test_missing_empty_replacement_multicall_and_structured_sources(tmp_path, monkeypatch):
    first = {"task_id": "task_001", "checklist": [{
        "criterion": "first criterion", "sources": [{"document": "paper.md", "quote": "paper text"}],
    }]}
    replacement = {"task_id": "task_001", "checklist": [{
        "criterion": "replacement", "sources": [{"document": "addendum.md", "quote": "addendum text"}],
    }]}
    responses = iter([
        _response(_call("finish_plan", {}, 1)),
        _response(_call("submit_task_checklist", first, 2),
                  _call("submit_task_checklist", {"task_id": "task_002", "checklist": []}, 3)),
        _response(_call("submit_task_checklist", {"task_id": "task_002", "checklist": []}, 4)),
        _response(_call("submit_task_checklist", replacement, 5)),
        _response(_call("finish_plan", {}, 6)),
    ])

    async def fake_request(*, completer, messages):
        assert completer is config
        assert [message["role"] for message in messages[:3]] == ["system", "user", "user"]
        assert messages[0]["content"] == CHECKLIST_ENRICHMENT_PROMPT
        assert "CHECKLIST FOR EACH SUBTASK" not in messages[0]["content"]
        assert "fixed legacy outline" in messages[1]["content"]
        assert messages[2]["content"] == CHECKLIST_ENRICHMENT_REQUEST
        assert "CHECKLIST FOR EACH SUBTASK" in messages[2]["content"]
        assert "TOOL WORKFLOW" in messages[2]["content"]
        if any(message.get("tool_call_id") == "call-1" for message in messages):
            assert "MISSING_TASKS" in messages[-1]["content"] or len(messages) > 4
        return next(responses)

    config = _Config()
    monkeypatch.setattr("research_agent.plan_enrichment_agent.make_resilient_completer_request", fake_request)
    store = PlanStore(tmp_path / "run")
    plan = asyncio.run(enrich_legacy_outline_agent(
        config=config, outline=_outline(), workspace_root=_workspace(tmp_path), store=store,
    ))
    assert plan.subtasks[0].checklist[0].criterion == "replacement"
    assert plan.subtasks[0].checklist[0].sources[0].document == "addendum.md"
    assert plan.subtasks[1].checklist == []
    assert plan.subtasks[0].name == "A"
    history = json.loads((store.run_dir / "checklist_enrichment_state.json").read_text())
    skipped = next(message for message in history["messages"] if message.get("tool_call_id") == "call-3")
    assert json.loads(skipped["content"])["code"] == "MULTIPLE_TOOL_CALLS_SKIPPED"


def test_resume_and_invalid_submission_does_not_mutate(tmp_path, monkeypatch):
    responses = iter([
        _response(_call("submit_task_checklist", {"task_id": "task_001", "checklist": []}, 1)),
        _response(_call("submit_task_checklist", {"task_id": "task_001", "checklist": [{
            "criterion": "bad", "sources": [{"document": "old-outline", "quote": "text"}],
        }]}, 2)),
        _response(_call("submit_task_checklist", {"task_id": "task_002", "checklist": []}, 3)),
        _response(_call("finish_plan", {}, 4)),
    ])
    config = _Config()

    async def fake_request(*, completer, messages):
        return next(responses)

    monkeypatch.setattr("research_agent.plan_enrichment_agent.make_resilient_completer_request", fake_request)
    store = PlanStore(tmp_path / "run")
    workspace = _workspace(tmp_path)
    with pytest.raises(RuntimeError, match="step limit"):
        asyncio.run(enrich_legacy_outline_agent(
            config=config, outline=_outline(), workspace_root=workspace, store=store, max_steps=2,
        ))
    state = json.loads((store.run_dir / "checklist_enrichment_state.json").read_text())
    assert state["submitted"] == {"task_001": []}
    assert state["quote_rejection_used_task_ids"] == []
    assert "INVALID_CHECKLIST" in state["messages"][-1]["content"]
    plan = asyncio.run(enrich_legacy_outline_agent(
        config=config, outline=_outline(), workspace_root=workspace, store=store,
    ))
    assert [task.checklist for task in plan.subtasks] == [[], []]
    assert store.load_plan() == plan


def test_quote_validation_rejects_once_then_skips_on_resume(tmp_path, monkeypatch):
    def submit(quote: str, document: str, number: int):
        return _response(_call("submit_task_checklist", {
            "task_id": "task_001", "checklist": [{
                "criterion": "check source", "sources": [{"document": document, "quote": quote}],
            }],
        }, number))

    responses = iter([
        submit("paper text", "paper.md", 1),
        submit("addendum text", "paper.md", 2),
        submit("paper ... text", "paper.md", 3),
        submit("PAPER \n TEXT", "paper.md", 4),
        _response(_call("submit_task_checklist", {"task_id": "task_002", "checklist": []}, 5)),
        _response(_call("finish_plan", {}, 6)),
    ])

    async def fake_request(*, completer, messages):
        return next(responses)

    monkeypatch.setattr("research_agent.plan_enrichment_agent.make_resilient_completer_request", fake_request)
    config = _Config()
    store = PlanStore(tmp_path / "run")
    workspace = _workspace(tmp_path)
    with pytest.raises(RuntimeError, match="step limit"):
        asyncio.run(enrich_legacy_outline_agent(
            config=config, outline=_outline(), workspace_root=workspace, store=store, max_steps=3,
        ))
    state = json.loads((store.run_dir / "checklist_enrichment_state.json").read_text())
    assert state["submitted"]["task_001"][0]["sources"][0]["quote"] == "paper ... text"
    assert state["quote_rejection_used_task_ids"] == ["task_001"]
    errors = [json.loads(message["content"]) for message in state["messages"]
              if message.get("role") == "tool" and message.get("tool_call_id") in {"call-2", "call-3"}]
    assert errors[0]["code"] == "QUOTE_NOT_FOUND"
    assert errors[1]["ok"] is True
    assert errors[1]["quote_existence_checked"] is False
    assert errors[0]["details"]["unmatched_sources"] == [{
        "item_index": 1, "source_index": 1, "document": "paper.md",
        "quote_found_in_other_documents": ["addendum.md"],
    }]
    plan = asyncio.run(enrich_legacy_outline_agent(
        config=config, outline=_outline(), workspace_root=workspace, store=store,
    ))
    assert plan.subtasks[0].checklist[0].sources[0].quote == "PAPER \n TEXT"
    assert plan.subtasks[1].checklist == []
    history = json.loads((store.run_dir / "checklist_enrichment_state.json").read_text())
    assert history["quote_rejection_used_task_ids"] == ["task_001"]
    resumed_result = next(json.loads(message["content"]) for message in history["messages"]
                          if message.get("tool_call_id") == "call-4")
    assert resumed_result["quote_existence_checked"] is False


def test_unmatched_quote_feedback_identifies_markdown_and_wrong_document():
    documents = {
        "paper.md": "The model uses an algorithm.",
        "addendum.md": "Additional rules.",
        "instructions.txt": "Your repository should be located at `/home/submission/`.",
    }
    normalized = {name: _normalized_quote(value) for name, value in documents.items()}
    item = SubmitInput.model_validate({
        "task_id": "task_001",
        "checklist": [{"criterion": "check", "sources": [
            {"document": "instructions.txt", "quote": "Your repository should be located at /home/submission/"},
            {"document": "addendum.md", "quote": "The model uses an algorithm."},
        ]}],
    })
    errors = _unmatched_sources(item, normalized, documents)
    assert errors[0]["nearby_document_line"] == documents["instructions.txt"]
    assert errors[1]["quote_found_in_other_documents"] == ["paper.md"]


def test_each_task_has_its_own_single_quote_rejection(tmp_path, monkeypatch):
    def bad_submission(task_id: str, number: int):
        return _response(_call("submit_task_checklist", {
            "task_id": task_id,
            "checklist": [{"criterion": "check", "sources": [
                {"document": "paper.md", "quote": "not in this paper"},
            ]}],
        }, number))

    responses = iter([
        bad_submission("task_001", 1),
        bad_submission("task_002", 2),
        bad_submission("task_001", 3),
        bad_submission("task_002", 4),
        _response(_call("finish_plan", {}, 5)),
    ])

    async def fake_request(*, completer, messages):
        return next(responses)

    monkeypatch.setattr("research_agent.plan_enrichment_agent.make_resilient_completer_request", fake_request)
    store = PlanStore(tmp_path / "run")
    plan = asyncio.run(enrich_legacy_outline_agent(
        config=_Config(), outline=_outline(), workspace_root=_workspace(tmp_path), store=store,
    ))
    assert [len(task.checklist) for task in plan.subtasks] == [1, 1]
    history = json.loads((store.run_dir / "checklist_enrichment_state.json").read_text())
    assert history["quote_rejection_used_task_ids"] == ["task_001", "task_002"]
    results = [json.loads(message["content"]) for message in history["messages"]
               if message.get("role") == "tool"]
    assert [result.get("code") for result in results] == [
        "QUOTE_NOT_FOUND", "QUOTE_NOT_FOUND", None, None, None,
    ]
    assert [result["quote_existence_checked"] for result in results[2:4]] == [False, False]
