"""Pin the copied NoChecker contract and exercise the real phase boundary offline."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from research_agent.context import MessageWindow
from research_agent.openai_oauth_completer import OpenAIOAuthTurnCompleterConfig
from research_agent.prompts import (
    FULL_REVIEW_EXECUTION_PROMPT, NO_CHECKER_EXECUTION_PROMPT, PLANNING_PROMPT,
    render_no_checker_plan_message, render_workspace_prompt,
)
from research_agent.runtime_tools import protocol_tools
from research_agent.session import ResearchAgentSession
from research_agent.solver import ResearchAgentSolver, _PaperBenchSnapshotter

from test_full_review_flow import ISSUE, make_manager


CONTRACT = json.loads((Path(__file__).parent / "fixtures/nochecker_contract.json").read_text())


def test_phase_one_matches_standalone_nochecker_contract(tmp_path):
    assert hashlib.sha256(NO_CHECKER_EXECUTION_PROMPT.encode()).hexdigest() == CONTRACT["execution_prompt_sha256"]
    assert hashlib.sha256(PLANNING_PROMPT.encode()).hexdigest() == CONTRACT["planning_prompt_sha256"]
    manager, _, _ = make_manager(tmp_path, [], rounds=2)
    assert [dict(t.get_oai_tool_call()) for t in protocol_tools(manager)] == CONTRACT["tools"]
    assert manager.tool_schemas() == CONTRACT["state_machine_tools"]
    assert manager.readonly_workspace.instructions_file == tmp_path / "workspace/instructions.txt"
    assert manager.readonly_workspace.checker_instructions_file is None
    items = [{"id": f"task_{i:03}", "name": f"name_{i}", "description": f"scope_{i}",
              "status": "uncompleted"} for i in range(1, 4)]
    for count, expected in enumerate(CONTRACT["plan_views"]):
        for i, item in enumerate(items):
            item["status"] = "completed" if i < count else "uncompleted"
        assert render_no_checker_plan_message(items) == expected


def test_first_stage_gates_and_feedback_match_nochecker(tmp_path):
    async def exercise():
        manager, checker, _ = make_manager(tmp_path, [], rounds=2)
        tools = {t.name(): t for t in protocol_tools(manager)}
        blocked = await tools["skip_subtask"].execute(None, subtask_id="task_001", reason="blocked")
        assert blocked == (
            "Please continue trying to complete subtask task_001. It has received "
            "0 valid Check Agent rejection(s); skip_subtask is allowed only after "
            "at least 3 valid Check Agent rejections. The subtask remains uncompleted."
        )
        assert not (await manager.final_submit()).accepted
        result = await tools["submit_subtask"].execute(
            None, subtask_id="task_001", reproduction_description="implemented")
        assert result == "The corresponding task in the reproduction plan has been changed to completed: task_001."
        assert manager.store.failure_count("task_001") == 0 and not checker.seen
        final = json.loads(await tools["final_submit"].execute(None))
        assert final == {"accepted": True, "snapshot_path": "canonical.tar.gz", "errors": []}
    asyncio.run(exercise())


def test_real_agent_loop_switches_only_after_final_submit(tmp_path, monkeypatch):
    async def exercise():
        manager, checker, canonical = make_manager(tmp_path, [[ISSUE]], rounds=1)
        session = ResearchAgentSession(store=manager.store, completer=None)
        session.window = MessageWindow(
            {"role": "system", "content": render_workspace_prompt(NO_CHECKER_EXECUTION_PROMPT, "/home")},
            render_no_checker_plan_message(manager.store.render()),
        )
        tools = protocol_tools(manager)
        config = OpenAIOAuthTurnCompleterConfig(
            credentials_dir=str(tmp_path / "oauth"), basicagent_tools=tools)
        calls = [
            ("skip_subtask", {"subtask_id": "task_001", "reason": "blocked"}),
            ("final_submit", {}),
            ("submit_subtask", {"subtask_id": "task_001", "reproduction_description": "done"}),
            (None, {}),
            ("final_submit", {}),
            ("respond_to_review", {"issue_id": "issue_1", "decision": "valid_issue",
                                   "reason": "fixed", "evidence": "submission/source.py"}),
            ("final_submit", {}),
        ]
        requests, completers = [], []
        cache = {"retained": [{"type": "reasoning", "id": "reasoning-test"}]}

        async def request(_request, *, completer, messages, **kwargs):
            index = len(requests)
            # Save actual provider input, including schema conversion and system role mapping.
            requests.append(copy.deepcopy(completer._request_body(messages)))
            completers.append(completer)
            completer._reasoning_items = cache
            name, args = calls[index]
            if name is None:
                return SimpleNamespace(response_messages=[{"role": "assistant", "content": "continue"}],
                                       tool_calls=[])
            call_id = f"call-{index}"
            return SimpleNamespace(response_messages=[{"role": "assistant", "content": None,
                "tool_calls": [{"id": call_id, "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args)}}]}],
                tool_calls=[SimpleNamespace(name=name, arguments=args, call_id=call_id)])

        monkeypatch.setattr("research_agent.solver.audited_request", request)
        monkeypatch.setattr("research_agent.solver._trim_execution_context", lambda **kwargs: (0, 0, 0))
        await ResearchAgentSolver(max_steps=15)._execute_agent_loop(
            computer=None, session=session, manager=manager, tools=tools, completer_config=config)
        assert len(requests) == len(calls) and len(checker.seen) == 1
        expected_system = render_workspace_prompt(NO_CHECKER_EXECUTION_PROMPT, "/home")
        for body in requests[:5]:
            assert body["input"][0]["content"] == expected_system
            assert body["tools"][0]["tools"] == CONTRACT["tools"]
        assert requests[5]["input"][0]["content"] == render_workspace_prompt(
            FULL_REVIEW_EXECUTION_PROMPT, "/home")
        review_names = {tool["name"] for tool in requests[5]["tools"][0]["tools"]}
        assert "respond_to_review" in review_names and "skip_subtask" not in review_names
        assert any(item.get("type") == "function_call_output" and
                   json.loads(item.get("output", "{}") or "{}").get("accepted") is True
                   for item in requests[5]["input"] if item.get("call_id") == "call-4")
        assert all(c is completers[0] for c in completers)
        assert session.review_started and manager.state.phase == "evaluation"
        assert canonical.calls == 2
    asyncio.run(exercise())


def test_review_switch_keeps_encrypted_reasoning_cache(tmp_path):
    from research_agent.solver import _activate_review_protocol

    manager, _, _ = make_manager(tmp_path, [], rounds=2)
    session = ResearchAgentSession(store=manager.store, completer=None, workspace_path="/custom")
    session.window = MessageWindow({"role": "system", "content": "initial"},
                                   render_no_checker_plan_message(manager.store.render()))
    session.window.append({"role": "assistant", "content": "retained history"})
    tools = protocol_tools(manager)
    config = OpenAIOAuthTurnCompleterConfig(
        credentials_dir=str(tmp_path / "oauth"), basicagent_tools=tools)
    completer = config.build()
    reasoning = completer._reasoning_items
    reasoning["call-old"] = [{"type": "reasoning", "encrypted_content": "opaque"}]
    call_items = completer._call_items
    call_items["call-old"] = {"type": "function_call"}
    manager.state.phase = "review_response"
    _activate_review_protocol(session=session, manager=manager, completer=completer,
                              completer_config=config, tools=tools)
    assert completer._reasoning_items is reasoning and "call-old" in reasoning
    assert completer._call_items is call_items and "call-old" in call_items
    assert session.window.history == [{"role": "assistant", "content": "retained history"}]
    assert "/custom/submission" in session.window.messages[0]["content"]
    assert completer.wait_for_account_usage_limit is True


def test_phase_one_quota_behavior_matches_nochecker(tmp_path, monkeypatch):
    async def exercise():
        config = OpenAIOAuthTurnCompleterConfig(credentials_dir=str(tmp_path))
        completer = config.build()
        failure = completer._failure("subscription_sharing_usage_limit_exceeded")
        success = SimpleNamespace(output=[SimpleNamespace(type="message", content=[
            SimpleNamespace(type="output_text", text="OK")])], usage=None)
        completer._complete_with_retries = AsyncMock(side_effect=[failure, failure, success])
        wait = AsyncMock()
        monkeypatch.setattr("research_agent.openai_oauth_completer.asyncio.sleep", wait)
        result = await completer.async_completion([{"role": "user", "content": "continue"}])
        assert result.output_messages[0].content == "OK"
        assert completer._complete_with_retries.call_count == 3
        assert [call.args[0] for call in wait.call_args_list] == [300.0, 300.0]
    asyncio.run(exercise())


def test_initial_snapshot_does_not_mark_review_run_done(tmp_path, monkeypatch):
    async def exercise():
        task = SimpleNamespace(run_dir=str(tmp_path), run_group_id="test", runs_dir=str(tmp_path), run_id="test")
        upload = AsyncMock()
        status = AsyncMock()
        monkeypatch.setattr("paperbench.solvers.upload.upload_heavy_logs", upload)
        monkeypatch.setattr("paperbench.solvers.upload.upload_status", status)
        monkeypatch.setattr("research_agent.solver.bf.glob", lambda _: ["submission.tar.gz"])
        snapshotter = _PaperBenchSnapshotter(computer=None, task=task, start_time=0)
        assert await snapshotter.freeze_for_review() == "submission.tar.gz"
        status.assert_not_called()
        assert await snapshotter.freeze_submission() == "submission.tar.gz"
        assert upload.call_count == 2 and status.call_args.kwargs["status"] == "done"
    asyncio.run(exercise())
