"""Offline monitor contracts using Codex notifications and temporary run trees."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

SPEC = importlib.util.spec_from_file_location("codex_monitor_server", Path(__file__).parents[1] / "server.py")
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def log(path, *events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def notification(method, item, milliseconds=1791853200000):
    return {"method": method, "emittedAtMs": milliseconds, "params": {"item": item}}


class CodexMonitorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "example-unique"
        self.output = self.run / "output"
        self.workspace = self.run / "home"
        (self.workspace / "submission").mkdir(parents=True)
        save(self.run / "launch.json", {"paper_id": "example", "model": "gpt-6-luna",
             "workspace_host_path": str(self.workspace), "output_host_path": str(self.output),
             "container_name": "codex-example-unique", "runtime": "docker", "skill_count": 3,
             "created_at_utc": "2026-10-13T01:00:00+00:00", "smoke_test": False})
        save(self.run / "host_status.json", {"phase": "running"})
        save(self.output / "status.json", {"phase": "execution"})
        save(self.output / "protocol/plan.json", {"subtasks": [
             {"id": "task_001", "name": "First", "description": "Implement method"},
             {"id": "task_002", "name": "Second", "description": "Documentation"}]})
        save(self.output / "protocol/plan_state.json", {"statuses": {
             "task_001": "uncompleted", "task_002": "uncompleted"}})
        log(self.output / "protocol/tool_calls.jsonl", {
            "timestamp": "2026-10-13T01:01:00+00:00", "event": "tool_call", "tool": "submit_subtask",
            "arguments": {"subtask_id": "task_001", "reproduction_description": "First revision"}}, {
            "timestamp": "2026-10-13T01:03:00+00:00", "event": "tool_call", "tool": "submit_subtask",
            "arguments": {"subtask_id": "task_001", "reproduction_description": "Second revision"}})
        log(self.output / "protocol/checker_calls.jsonl", {
            "timestamp": "2026-10-13T01:01:01+00:00", "event": "checker_started", "task_id": "task_001", "attempt": 1}, {
            "timestamp": "2026-10-13T01:02:00+00:00", "event": "checker_response", "task_id": "task_001",
            "report": {"accepted": False, "summary": "Missing method", "issues": [
                {"description": "Incomplete", "evidence": "solution.py"}]}}, {
            "timestamp": "2026-10-13T01:03:01+00:00", "event": "checker_started", "task_id": "task_001", "attempt": 1})
        tool = {"type": "fileChange", "id": "call-1", "status": "inProgress",
                "changes": [{"path": "solution.py", "kind": "add", "diff": "source text must not appear"}]}
        log(self.output / "main.jsonl",
            notification("item/started", tool),
            notification("item/completed", {**tool, "status": "completed"}),
            notification("item/started", {"type": "commandExecution", "command": "python -m py_compile solution.py",
                                          "cwd": "/home/submission", "status": "inProgress"}),
            notification("item/completed", {"type": "commandExecution", "status": "completed",
                                            "exitCode": 0, "aggregatedOutput": ""}),
            notification("item/completed", {"type": "agentMessage", "text": "Working on the first task"}))
        for serial in (1, 2):
            log(self.output / f"checker/task_001-review-{serial}.jsonl", notification("item/started", {
                "type": "commandExecution", "command": f"sed -n '1,80p' /home/submission/revision{serial}.py",
                "cwd": "/home", "status": "inProgress"}))
        self.monitor = server.Monitor(self.root)

    def refresh(self, active=True):
        with patch.object(self.monitor, "_active_containers", return_value={"codex-example-unique"} if active else set()):
            return self.monitor.refresh()

    def test_codex_events_and_per_review_evidence(self):
        value = self.refresh()
        self.assertEqual(value["run"]["phase"], "Checker 验收")
        self.assertTrue(value["run"]["running"])
        self.assertEqual(value["run"]["skill_count"], 3)
        self.assertEqual(value["protocol"]["submit_task_calls"], 2)
        self.assertTrue(value["tasks"][1]["hidden_from_agent"])
        attempts = value["checker_trace"]["attempts"]
        self.assertEqual([item["reproduction_description"] for item in attempts], ["First revision", "Second revision"])
        self.assertEqual(attempts[0]["result"], "rejected")
        self.assertIsNone(attempts[1]["result"])
        self.assertEqual(attempts[1]["reads"], ["sed -n '1,80p' /home/submission/revision2.py"])
        self.assertEqual(sum(item["event"] == "tool_call" for item in value["main_trace"]), 2)
        self.assertIn("write solution.py", str(value["main_trace"]))
        self.assertNotIn("source text must not appear", str(value))
        self.assertTrue(all(item["time_bj"] for item in value["main_trace"]))

    def test_incremental_tail_partial_line_and_retry(self):
        first = self.refresh()
        event = {"method": "error", "emittedAtMs": 1791853300000,
                 "params": {"willRetry": True, "error": {"message": "temporary failure sk-testprivate"}}}
        encoded = json.dumps(event)
        path = self.output / "main.jsonl"
        with path.open("a") as stream:
            stream.write(encoded[:20])
        self.assertEqual(len(self.refresh()["main_trace"]), len(first["main_trace"]))
        with path.open("a") as stream:
            stream.write(encoded[20:] + "\n")
        value = self.refresh()
        self.assertTrue(value["main_trace"][-1]["retrying"])
        self.assertNotIn("sk-testprivate", str(value))
        self.assertEqual(value["diagnostics"]["main_bad_lines"], 0)

    def test_grading_failure_is_not_success_or_zero_score(self):
        save(self.output / "status.json", {"phase": "complete", "accepted": True})
        save(self.run / "host_status.json", {"phase": "grading_failed", "updated_at": "2026-10-13T01:10:00+00:00"})
        value = self.refresh(False)
        self.assertEqual(value["run"]["phase"], "评分失败")
        self.assertTrue(value["run"]["markers"]["error"])
        self.assertEqual(value["run"]["elapsed_seconds"], 600)
        self.assertIsNone(value["evaluation"]["score_percent"])
        self.assertFalse(value["run"]["running"])

    def test_grading_process_and_official_score(self):
        save(self.run / "host_status.json", {"phase": "grading"})
        save(self.run / "evaluation/runs/group/example-run/grade.json", {
            "paperbench_result": {"judge_output": {"score": 0.8433, "num_invalid_leaf_nodes": 2}}})
        with patch.object(self.monitor, "_grading_process", return_value=(1234, 1000)):
            value = self.refresh(False)
        self.assertTrue(value["run"]["running"])
        self.assertEqual(value["run"]["phase"], "评分中")
        self.assertEqual(value["evaluation"]["score_percent"], 84.33)
        self.assertEqual(value["evaluation"]["reports"][0]["invalid_leaf_nodes"], 2)

    def test_smoke_and_debug_runs_are_excluded(self):
        save(self.root / "debug/other.json", {})
        save(self.root / "smoke/launch.json", {"paper_id": "example", "smoke_test": True})
        self.assertEqual(len(self.refresh()["runs"]), 1)
        with self.assertRaises(ValueError):
            server.Monitor(self.root, self.root.parent)

    def test_http_endpoints_and_static_boundary(self):
        self.refresh()
        handler = type("TestHandler", (server.Handler,), {"monitor": self.monitor})
        http = server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{http.server_port}"
            for route in ("/api/health", "/api/state", "/api/runs", "/api/files", "/api/trace?role=main", "/api/trace?role=checker"):
                with urlopen(base + route, timeout=3) as response:
                    self.assertTrue(json.load(response)["ok"])
            with urlopen(base, timeout=3) as response:
                self.assertIn("Codex Agent", response.read().decode())
            with self.assertRaises(HTTPError) as caught:
                urlopen(base + "/../server.py", timeout=3)
            self.assertEqual(caught.exception.code, 404)
            caught.exception.close()
        finally:
            http.shutdown()
            http.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
