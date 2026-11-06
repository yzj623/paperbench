import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from research_agent.audit import audited_request, snapshot
from research_agent.context import MessageWindow
from research_agent.storage import PlanStore, redact_sensitive


def test_snapshot_diff_and_credential_redaction(tmp_path):
    store = PlanStore(tmp_path / "logs")
    source = tmp_path / "submission"
    source.mkdir()
    (source / "code.py").write_text("value = 1\n")
    (source / ".env").write_text("API_KEY=never-store-me")
    snapshot(store, source, "before", "submit_subtask", {"subtask_id": "task_001"})
    (source / "code.py").write_text("value = 2\n")
    snapshot(store, source, "after", "submit_subtask", {"accepted": False})
    manifests = [json.loads(p.read_text()) for p in (store.run_dir / "audit/snapshots").glob("*.json")]
    second = next(x for x in manifests if x["phase"] == "after")
    assert second["previous_snapshot_id"]
    assert "+value = 2" in second["changes"][0]["diff"]
    assert ".env" in second["omitted"]
    assert "never-store-me" not in "".join(p.read_text() for p in (store.run_dir / "audit").rglob("*") if p.is_file())
    assert redact_sensitive({"access_token": "secret", "input_tokens": 12}) == {"access_token": "[REDACTED]", "input_tokens": 12}


def test_request_response_trim_and_no_behavior_change(tmp_path):
    store = PlanStore(tmp_path / "logs")
    window = MessageWindow({"role": "system", "content": "system"}, {"role": "user", "content": "plan"})
    window.audit_log = store.tool_log
    window.append({"role": "user", "content": "checker rejected: fix mask"})
    window.trim_oldest_unit()
    @dataclass
    class Response:
        response_messages: list
        usage: dict
    expected = Response([{"role": "assistant", "content": "ok"}], {"input_tokens": 12})
    async def request(**kwargs):
        assert kwargs["messages"] == window.messages
        return expected
    result = asyncio.run(audited_request(request, audit_log=store.tool_log, role="main",
        completer=SimpleNamespace(model="fixture", tools=[]), messages=window.messages))
    assert result is expected
    records = [json.loads(x) for x in (store.run_dir / "audit/events.jsonl").read_text().splitlines()]
    trim = next(x for x in records if x["event"] == "context_unit_removed")
    assert trim["removed_messages"][0]["content"] == "checker rejected: fix mask"
    response = next(x for x in records if x["event"] == "model_response")
    assert response["response"]["usage"]["input_tokens"] == 12
    assert response["request_id"] == next(x for x in records if x["event"] == "model_request")["request_id"]


def test_completion_retries_and_raw_usage_are_preserved(tmp_path):
    store = PlanStore(tmp_path / "logs")
    class Completer:
        tools = []
        calls = 0
        async def async_completion(self, conversation, **params):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("fixture malformed response")
            return {"output_messages": [{"role": "assistant", "content": "fixed"}],
                    "usage": {"input_tokens": 30}}
    completer = Completer()
    async def request(completer, messages):
        try:
            return await completer.async_completion(conversation=messages)
        except ValueError:
            messages.append({"role": "user", "content": "retry with valid response"})
            return await completer.async_completion(conversation=messages)
    result = asyncio.run(audited_request(request, audit_log=store.tool_log, role="checker",
        completer=completer, messages=[{"role": "user", "content": "review"}]))
    assert completer.calls == 2 and result["usage"]["input_tokens"] == 30
    records = [json.loads(x) for x in (store.run_dir / "audit/events.jsonl").read_text().splitlines()]
    attempts = [x for x in records if x["event"] == "completion_attempt"]
    assert [x["attempt"] for x in attempts] == [1, 2]
    assert attempts[-1]["messages"][-1]["content"] == "retry with valid response"
    assert any(x["event"] == "completion_attempt_error" for x in records)

