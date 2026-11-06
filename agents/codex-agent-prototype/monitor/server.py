#!/usr/bin/env python3
"""Read-only web monitor for the standalone PaperBench Codex harness."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import logging
import mimetypes
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
PROJECT = HERE.parent
BEIJING = timezone(timedelta(hours=8))
MAX_EVENTS = 1200
MAX_LINE_BYTES = 12 * 1024 * 1024
SECRET_RE = re.compile(r"(?i)(api[_-]?key|authorization|bearer|token)\s*[:=]\s*[^\s,}\]]+")
SECRET_VALUE_RE = re.compile(r"(?:sk-[A-Za-z0-9][A-Za-z0-9_.-]*|Bearer\s+\S+)", re.IGNORECASE)


def setup_logging() -> logging.Logger:
    log_dir = HERE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("codex-paperbench-monitor")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(log_dir / "monitor.log", maxBytes=2_000_000, backupCount=3)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def redact(value: object, limit: int = 1600) -> str:
    text = SECRET_RE.sub(r"\1=[REDACTED]", str(value).replace("\x00", ""))
    text = SECRET_VALUE_RE.sub("[REDACTED]", text)
    return text if len(text) <= limit else text[:limit] + f"… ({len(text) - limit} chars hidden)"


def time_epoch(value: object) -> float | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() if value else None
    except (TypeError, ValueError):
        return None


def iso_time(epoch: float | None, tz: timezone = BEIJING) -> str | None:
    return datetime.fromtimestamp(epoch, tz).isoformat() if epoch is not None else None


def safe_json(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return copy.deepcopy(default)


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


class JsonlTail:
    """Incrementally read append-only logs, retaining only relevant events."""

    def __init__(self, path: Path, accepted: set[str] | None = None):
        self.path = path
        self.accepted = accepted
        self.offset = 0
        self.identity: tuple[int, int] | None = None
        self.partial = b""
        # Keep authoritative protocol records for exact lifetime counters, while
        # bounding the much more verbose engine trace shown in the browser.
        self.events: collections.deque[dict] = collections.deque(
            maxlen=MAX_EVENTS if accepted == CODEX_EVENTS else None)
        self.bad_lines = 0

    def refresh(self) -> None:
        try:
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity != self.identity or stat.st_size < self.offset:
                self.offset, self.partial, self.bad_lines = 0, b"", 0
                self.events.clear()
            self.identity = identity
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                while chunk := stream.read(128 * 1024):
                    self.offset += len(chunk)
                    lines = (self.partial + chunk).split(b"\n")
                    self.partial = lines.pop()
                    if len(self.partial) > MAX_LINE_BYTES:
                        self.partial = b""
                        self.bad_lines += 1
                    for raw in lines:
                        if not raw.strip() or len(raw) > MAX_LINE_BYTES:
                            continue
                        try:
                            event = json.loads(raw)
                            if isinstance(event, dict) and (
                                self.accepted is None or event.get("method", event.get("event")) in self.accepted
                            ):
                                if self.accepted == CODEX_EVENTS:
                                    event = compact_codex_event(event)
                                if event is None:
                                    continue
                                self.events.append(event)
                        except (UnicodeDecodeError, ValueError):
                            self.bad_lines += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            LOGGER.warning("Cannot refresh %s: %s", self.path, exc)


CODEX_EVENTS = {"item/started", "item/completed", "error", "turn/completed"}
CHECKER_EVENTS = {"checker_started", "checker_response", "checker_response_invalid", "checker_call_error"}
TERMINAL_PHASES = {"complete", "failed", "interrupted", "grading_failed", "grading_interrupted",
                   "preparation_failed", "launch_failed", "submission_rejected", "smoke_complete"}


def compact_codex_event(event: dict) -> dict | None:
    """Normalize app-server notifications without retaining source-code payloads."""
    method = event.get("method")
    params = event.get("params") if isinstance(event.get("params"), dict) else {}
    item = params.get("item") if isinstance(params.get("item"), dict) else {}
    milliseconds = event.get("emittedAtMs", params.get("completedAtMs", params.get("startedAtMs")))
    stamp = iso_time(milliseconds / 1000, timezone.utc) if isinstance(milliseconds, (int, float)) else None
    base = {"timestamp": stamp, "time_bj": iso_time(time_epoch(stamp))}
    if method == "item/started" and item.get("type") == "dynamicToolCall":
        args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        allowed = {"path", "file", "cwd", "log_path", "cmd", "subtask_id", "reason",
                   "reproduction_description", "job_id", "query", "offset", "limit",
                   "start_line", "max_lines", "page", "tail_lines"}
        return {**base, "type": "tool_execution_start", "toolName": item.get("tool"),
                "args": {key: redact(value, 1800) for key, value in args.items() if key in allowed}}
    if method == "item/started" and item.get("type") == "commandExecution":
        return {**base, "type": "tool_execution_start", "toolName": "codex.commandExecution",
                "args": {"command": redact(item.get("command", ""), 1800),
                         "cwd": redact(item.get("cwd", ""), 300)}}
    if method == "item/started" and item.get("type") == "fileChange":
        changes = item.get("changes") if isinstance(item.get("changes"), list) else []
        return {**base, "type": "tool_execution_start", "toolName": "codex.fileChange",
                "args": {"paths": [redact(change.get("path", ""), 300)
                                    for change in changes if isinstance(change, dict)]}}
    if method == "item/completed":
        if item.get("type") == "agentMessage":
            return {**base, "type": "message_end", "message": {"role": "assistant",
                    "content": [{"type": "text", "text": redact(item.get("text", ""), 3000)}]}}
        if item.get("type") == "dynamicToolCall":
            content = item.get("contentItems") if isinstance(item.get("contentItems"), list) else []
            return {**base, "type": "tool_execution_end", "toolName": item.get("tool"),
                    "success": item.get("success"), "duration_ms": item.get("durationMs"),
                    "result": {"content": [{"text": redact(part.get("text", ""), 1800)}
                                           for part in content if isinstance(part, dict)]}}
        if item.get("type") == "commandExecution":
            return {**base, "type": "tool_execution_end", "toolName": "codex.commandExecution",
                    "success": item.get("status") == "completed" and item.get("exitCode") == 0,
                    "result": {"content": [{"text": redact(item.get("aggregatedOutput", ""), 1800)}]}}
        if item.get("type") == "fileChange":
            return {**base, "type": "tool_execution_end", "toolName": "codex.fileChange",
                    "success": item.get("status") == "completed"}
        if item.get("type") == "contextCompaction":
            return {**base, "type": "engine_status", "summary": "Codex 完成上下文压缩"}
    if method == "error":
        return {**base, "type": "engine_error", "summary": redact(params.get("error", "")),
                "retrying": bool(params.get("willRetry"))}
    if method == "turn/completed":
        turn = params.get("turn") if isinstance(params.get("turn"), dict) else {}
        if turn.get("status") != "completed" or turn.get("error"):
            return {**base, "type": "engine_error", "summary": redact(turn.get("error") or turn.get("status"))}
    return None



def read_evaluation(run: Path, *, grading_active: bool = False, host_phase: str | None = None) -> dict:
    """Read official reports only; never infer a missing/failed score as zero."""
    reports, invalid = [], []
    failed_report = False
    roots = [run / "evaluation", run / "grade"]
    for root in roots:
        if not root.is_dir() or not inside(root, run):
            continue
        for path in root.glob("runs/*/*/grade.json"):
            if not inside(path, run) or path.is_symlink():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("report is not an object")
                paper = value.get("paperbench_result", {})
                paper = paper if isinstance(paper, dict) else {}
                judge = paper.get("judge_output", {})
                judge = judge if isinstance(judge, dict) else {}
                if value.get("grading_failed") or paper.get("grading_failed") or ("judge_output" in paper and paper["judge_output"] is None):
                    failed_report = True
                    raise ValueError("grading failed")
                score = judge.get("score", value.get("score"))
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("missing or invalid score")
                modified = path.stat().st_mtime
                count = judge.get("num_invalid_leaf_nodes")
                reports.append({
                    "score": score, "score_percent": round(score * 100, 2),
                    "invalid_leaf_nodes": count, "graded_at": judge.get("graded_at"),
                    "modified_epoch": modified,
                    "graded_at_bj": datetime.fromtimestamp(modified, timezone(timedelta(hours=8))).isoformat(),
                    "path": path.relative_to(run).as_posix(),
                })
            except (OSError, ValueError, TypeError):
                invalid.append({"path": path.relative_to(run).as_posix(), "reason": "评分文件暂不可读、评分失败或分数无效"})
    reports.sort(key=lambda item: (item["modified_epoch"], item["path"]))
    latest = reports[-1] if reports else None
    if host_phase in {"grading_failed", "grading_interrupted"} or (host_phase in {"failed", "interrupted"} and any(root.is_dir() for root in roots)):
        status, label = "failed", "评分失败或中断"
    elif grading_active:
        status, label = "grading", "评分中"
    elif latest:
        status, label = "complete", "评分完成"
    elif failed_report:
        status, label = "failed", "评分失败"
    elif invalid:
        status, label = "unavailable", "评分结果异常或尚未写完"
    elif any(root.is_dir() for root in roots):
        status, label = "waiting", "暂无评分结果"
    else:
        status, label = "pending", "尚未评分"
    return {"status": status, "label": label,
            "score_percent": latest["score_percent"] if latest else None,
            "invalid_leaf_nodes": latest["invalid_leaf_nodes"] if latest else None,
            "reports": reports, "invalid_reports": invalid,
            "latest_report": latest}


def find_grading_process(run: Path) -> tuple[int | None, float | None]:
    needles = (str(run / "evaluation"), str(run / "grade"))
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            if not any(needle in command for needle in needles):
                continue
            if "evaluate_submission.sh" in command or "eval_entrypoint" in command:
                return int(entry.name), None
        except OSError:
            continue
    return None, None

class Monitor:
    def __init__(self, runs_root: Path, pinned_run: Path | None = None):
        self.root = runs_root.resolve(strict=True)
        self.pinned = pinned_run.resolve(strict=True) if pinned_run else None
        if self.pinned is not None and not inside(self.pinned, self.root):
            raise ValueError("CODEX_RUN_DIR must be under CODEX_RUNS_ROOT")
        self.lock = threading.RLock()
        self.caches: dict[Path, JsonlTail] = {}
        self.snapshot: dict = {}
        self.selected_root: Path | None = None
        self.stop_event = threading.Event()

    def _cache(self, path: Path, accepted: set[str] | None = None) -> JsonlTail:
        if path not in self.caches:
            self.caches[path] = JsonlTail(path, accepted)
        cache = self.caches[path]
        cache.refresh()
        return cache

    def _active_containers(self) -> set[str]:
        try:
            result = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                                    capture_output=True, text=True, timeout=3, check=False)
            return set(result.stdout.splitlines()) if result.returncode == 0 else set()
        except (OSError, subprocess.TimeoutExpired):
            return set()

    def _grading_process(self, run: Path) -> tuple[int | None, float | None]:
        needle = str(run / "evaluation")
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
                if "evaluate_submission.sh" not in command or needle not in command:
                    continue
                stat = (entry / "stat").read_text(encoding="utf-8").split()
                boot = float(Path("/proc/stat").read_text(encoding="utf-8").split("btime ", 1)[1].splitlines()[0])
                return int(entry.name), boot + int(stat[21]) / os.sysconf("SC_CLK_TCK")
            except (OSError, ValueError, IndexError):
                continue
        return None, None

    def _runs(self, containers: set[str]) -> list[dict]:
        found = []
        for run in self.root.iterdir():
            if not run.is_dir() or not inside(run, self.root):
                continue
            meta = safe_json(run / "launch.json", {})
            if not isinstance(meta, dict) or not meta.get("paper_id") or meta.get("smoke_test"):
                continue
            workspace = Path(str(meta.get("workspace_host_path", "")))
            output = Path(str(meta.get("output_host_path", "")))
            if not inside(workspace, run) or not inside(output, run):
                continue
            docker_name = str(meta.get("container_name", ""))
            host = safe_json(run / "host_status.json", {})
            active = docker_name in containers
            if isinstance(host, dict) and host.get("phase") == "grading":
                active = active or self._grading_process(run)[0] is not None
            found.append({"paper_id": meta["paper_id"], "run_dir": str(run),
                          "model": meta.get("model"), "active": active,
                          "started_epoch": (run / "launch.json").stat().st_mtime,
                          "modified_bj": iso_time(run.stat().st_mtime)})
        found.sort(key=lambda item: (not item["active"], -item["started_epoch"]))
        return found

    @staticmethod
    def _main_trace(events: list[dict]) -> list[dict]:
        output: list[dict] = []
        for event in events:
            kind = event.get("type")
            base = {"timestamp": event.get("timestamp"), "time_bj": event.get("time_bj")}
            message = event.get("message") if isinstance(event.get("message"), dict) else {}
            if kind == "message_end" and message.get("role") == "assistant":
                for part in message.get("content") or []:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text" and str(part.get("text", "")).strip():
                        output.append({**base, "event": "assistant", "tool": None, "task_id": None,
                                       "paths": [], "summary": redact(part["text"], 1400)})
                    if part.get("type") == "toolCall":
                        name = str(part.get("name", ""))
                        args = part.get("arguments") if isinstance(part.get("arguments"), dict) else {}
                        paths = [redact(args[key], 400) for key in ("path", "file", "cwd", "log_path") if key in args]
                        summary = (redact(args.get("cmd", ""), 900) if name == "bash" else
                                   f"write {redact(args.get('path', ''), 500)}" if name == "write_file" else
                                   redact({k: v for k, v in args.items() if k not in {"content", "env"}}, 900))
                        output.append({"event": "tool_call", "tool": name, "task_id": args.get("subtask_id"),
                                       "paths": paths, "summary": summary})
            elif kind == "tool_execution_start":
                name = str(event.get("toolName", ""))
                args = event.get("args") or {}
                paths = [redact(args[key], 400) for key in ("path", "file", "cwd", "log_path") if key in args]
                if name == "codex.fileChange":
                    paths = [redact(path, 400) for path in args.get("paths", [])]
                summary = (redact(args.get("command", ""), 900) if name == "codex.commandExecution" else
                           "write " + ", ".join(paths) if name == "codex.fileChange" else
                           redact(args.get("cmd", ""), 900) if name == "bash" else
                           f"write {redact(args.get('path', ''), 500)}" if name == "write_file" else
                           redact(args, 900))
                output.append({**base, "event": "tool_call", "tool": name,
                               "task_id": args.get("subtask_id"), "paths": paths, "summary": summary})
            elif kind == "tool_execution_end":
                result = event.get("result") if isinstance(event.get("result"), dict) else {}
                content = result.get("content") if isinstance(result.get("content"), list) else []
                rendered = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
                output.append({**base, "event": "tool_result", "tool": event.get("toolName"), "task_id": None,
                               "success": event.get("success"),
                               "paths": [], "summary": redact(rendered, 1200)})
            elif kind in {"engine_error", "engine_status"}:
                output.append({**base, "event": kind, "tool": None, "task_id": None, "paths": [],
                               "summary": redact(event.get("summary", ""), 1200),
                               "retrying": event.get("retrying", False)})
        return output[-400:]

    def _checker_trace(self, run: Path, tasks: dict[str, dict], events: list[dict],
                       main_events: list[dict]) -> dict:
        submissions = []
        for event in main_events:
            if event.get("event") == "tool_call" and event.get("tool") == "submit_subtask":
                args = event.get("arguments") or {}
                submissions.append((str(args.get("subtask_id")), time_epoch(event.get("timestamp")),
                                    redact(args.get("reproduction_description", ""))))
        attempts = []
        for event in events:
            kind = event.get("event")
            if kind == "checker_started":
                tid = str(event.get("task_id"))
                started = time_epoch(event.get("timestamp"))
                descriptions = [description for task_id, stamp, description in submissions
                                if task_id == tid and (started is None or stamp is None or stamp <= started)]
                attempts.append({"attempt": event.get("attempt"), "task_id": tid,
                                 "task_name": tasks.get(tid, {}).get("name"),
                                 "reproduction_description": descriptions[-1] if descriptions else "",
                                 "started": event.get("timestamp"),
                                 "started_bj": iso_time(time_epoch(event.get("timestamp"))),
                                 "reads": [], "result": None, "reason": None, "issues": [],
                                 "format_error": False, "error": None, "duration_seconds": None})
                continue
            if not attempts:
                continue
            current = attempts[-1]
            if kind == "checker_response":
                report = event.get("report") if isinstance(event.get("report"), dict) else {}
                current["result"] = "accepted" if report.get("accepted") else "rejected"
                current["reason"] = redact(report.get("summary", ""))
                issues = report.get("issues") if isinstance(report.get("issues"), list) else []
                current["issues"] = [
                    {"description": redact(issue.get("description", ""), 1600),
                     "evidence": redact(issue.get("evidence", ""), 1600)}
                    for issue in issues[:30] if isinstance(issue, dict)
                ]
            elif kind == "checker_response_invalid":
                current["format_error"] = True
                current["error"] = redact(event.get("details", ""))
            elif kind == "checker_call_error":
                current["error"] = redact(event.get("error", ""))
            if kind in {"checker_response", "checker_response_invalid", "checker_call_error"}:
                end = time_epoch(event.get("timestamp")); start = time_epoch(current.get("started"))
                if end and start:
                    current["duration_seconds"] = round(end - start, 3)
        # Protocol attempts and Codex checker files share the same per-task order.
        checker_dir = run / "checker"
        by_task: dict[str, list[Path]] = {}
        if checker_dir.is_dir():
            for path in checker_dir.glob("*-review-*.jsonl"):
                match = re.fullmatch(r"(.+)-review-(\d+)\.jsonl", path.name)
                if match:
                    by_task.setdefault(match.group(1), []).append(path)
        for paths in by_task.values():
            paths.sort(key=lambda path: int(path.stem.rsplit("-", 1)[1]))
        indices: collections.Counter[str] = collections.Counter()
        for attempt in attempts:
            tid = attempt["task_id"]
            position = indices[tid]
            indices[tid] += 1
            paths = by_task.get(tid, [])
            if position >= len(paths):
                continue
            events_for_attempt = self._cache(paths[position], CODEX_EVENTS).events
            for event in events_for_attempt:
                if event.get("type") == "tool_execution_start" and event.get("toolName") in {
                    "read_file", "list_directory", "search_file"
                }:
                    path = (event.get("args") or {}).get("path")
                    if path:
                        attempt["reads"].append(redact(path, 500))
                elif event.get("type") == "tool_execution_start" and event.get("toolName") == "codex.commandExecution":
                    command = (event.get("args") or {}).get("command")
                    if command:
                        attempt["reads"].append(redact(command, 500))
        return {"message": None if attempts else "尚未调用 Checker", "attempts": attempts[-80:],
                "latest_by_task": {attempt["task_id"]: attempt for attempt in attempts}}

    def _files(self, run: Path, workspace: Path) -> list[dict]:
        roots = [(workspace / "submission", "home/submission"),
                 (run / "output/frozen/submission", "output/frozen/submission"),
                 (run / "evaluation", "evaluation")]
        files: list[dict] = []
        for base, label in roots:
            if not base.is_dir() or not inside(base, self.root):
                continue
            for directory, dirs, names in os.walk(base):
                dirs[:] = [name for name in dirs if name not in {".git", "__pycache__", ".venv", "node_modules"}]
                for name in names:
                    path = Path(directory) / name
                    if path.is_symlink():
                        continue
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    files.append({"path": f"{label}/{path.relative_to(base).as_posix()}",
                                  "name": name, "size": stat.st_size,
                                  "modified_bj": iso_time(stat.st_mtime)})
                    if len(files) >= 3000:
                        return files
        return sorted(files, key=lambda item: item["path"])

    def _evaluation(self, run: Path) -> dict:
        return read_evaluation(run)

    def refresh(self) -> dict:
        with self.lock:
            containers = self._active_containers()
            runs = self._runs(containers)
            selected = (next((item for item in runs if item["run_dir"] == str(self.pinned)), None)
                        if self.pinned is not None else None)
            if selected is None and self.pinned is None:
                selected = next((item for item in runs if item["active"]), None)
                if selected is None and self.selected_root is not None:
                    selected = next((item for item in runs if item["run_dir"] == str(self.selected_root)), None)
                selected = selected or (runs[0] if runs else None)
            if selected is None:
                self.snapshot = {"ok": True, "message": "没有 Codex 实验", "runs": runs}
                return copy.deepcopy(self.snapshot)
            run = Path(selected["run_dir"])
            self.selected_root = run
            meta = safe_json(run / "launch.json", {})
            host = safe_json(run / "host_status.json", {})
            output_dir = Path(meta["output_host_path"])
            state = safe_json(output_dir / "status.json", {})
            host = host if isinstance(host, dict) else {}
            state = state if isinstance(state, dict) else {}
            workspace = Path(meta["workspace_host_path"])
            protocol_dir = output_dir / "protocol"
            plan = safe_json(protocol_dir / "plan.json", {"subtasks": []})
            plan_state = safe_json(protocol_dir / "plan_state.json", {"statuses": {}})
            tasks_list = plan.get("subtasks", []) if isinstance(plan, dict) else []
            statuses = plan_state.get("statuses", {}) if isinstance(plan_state, dict) else {}
            tasks_list = tasks_list if isinstance(tasks_list, list) else []
            statuses = statuses if isinstance(statuses, dict) else {}
            task_map = {task.get("id"): task for task in tasks_list if isinstance(task, dict)}
            calls = list(self._cache(protocol_dir / "tool_calls.jsonl").events)
            checker_events = list(self._cache(protocol_dir / "checker_calls.jsonl", CHECKER_EVENTS).events)
            state_events = list(self._cache(protocol_dir / "events.jsonl").events)
            main_events = list(self._cache(output_dir / "main.jsonl", CODEX_EVENTS).events)
            checker = self._checker_trace(output_dir, task_map, checker_events, calls)
            count: collections.Counter[str] = collections.Counter()
            submissions: collections.Counter[str] = collections.Counter()
            for event in calls:
                if event.get("event") == "tool_call" and event.get("tool") in {
                    "submit_subtask", "skip_subtask", "final_submit"
                }:
                    count[event["tool"]] += 1
                    if event["tool"] == "submit_subtask":
                        submissions[(event.get("arguments") or {}).get("subtask_id")] += 1
            current_index = next((index for index, task in enumerate(tasks_list)
                                  if statuses.get(task.get("id"), "uncompleted") == "uncompleted"),
                                 len(tasks_list))
            last_checker = checker.pop("latest_by_task")
            tasks = []
            for index, task in enumerate(tasks_list):
                tid = task.get("id")
                last = last_checker.get(tid, {})
                tasks.append({"task_id": tid, "name": task.get("name"),
                              "description": task.get("description"), "skill_ids": [],
                              "status": statuses.get(tid, "uncompleted"),
                              "submit_attempts": submissions[tid],
                              "checker_result": last.get("result"),
                              "checker_reason": last.get("reason"),
                              "hidden_from_agent": index > current_index})
            docker_active = str(meta.get("container_name")) in containers
            grader_pid, _grader_start = self._grading_process(run) if host.get("phase") == "grading" else (None, None)
            running = docker_active or grader_pid is not None
            phase = host.get("phase")
            if phase == "grading" and grader_pid is None:
                phase = "评分中断或状态待确认"
            phase_names = {"preparing": "准备中", "running": "Main Agent 实现",
                           "grading": "评分中", "complete": "已完成", "failed": "失败",
                           "interrupted": "已中断", "grading_failed": "评分失败",
                           "grading_interrupted": "评分已中断", "launch_failed": "启动失败",
                           "preparation_failed": "准备失败", "submission_rejected": "最终提交未通过"}
            if docker_active:
                phase = ("Checker 验收" if checker["attempts"] and
                         checker["attempts"][-1]["result"] is None and not checker["attempts"][-1]["error"] else
                         "Main Agent 实现" if (protocol_dir / "plan.json").is_file() else "规划")
            else:
                phase = phase_names.get(phase, str(phase or state.get("phase", "未知")))
            evaluation = read_evaluation(run, grading_active=grader_pid is not None, host_phase=host.get("phase"))
            if not running and evaluation["status"] == "complete" and host.get("phase") not in TERMINAL_PHASES - {"complete", "smoke_complete"}:
                phase = "评分完成"
            stats = {"submit_task_calls": count["submit_subtask"],
                     "submit_passed": sum(task["status"] == "completed" for task in tasks),
                     "submit_rejected": sum(event.get("event") == "checker_response" and
                                            isinstance(event.get("report"), dict) and
                                            not event["report"].get("accepted") for event in checker_events),
                     "skip_task_calls": count["skip_subtask"],
                     "skipped_tasks": sum(task["status"] == "skipped" for task in tasks),
                     "final_submit_calls": count["final_submit"],
                     "final_accepted": int(state.get("phase") == "complete")}
            timestamps = [time_epoch(e.get("timestamp")) for e in calls + checker_events + state_events]
            watched = [output_dir / "main.jsonl", output_dir / "planner.jsonl", output_dir / "status.json", run / "host_status.json"]
            timestamps += [path.stat().st_mtime for path in watched if path.is_file()]
            start = time_epoch(meta.get("created_at_utc")) or (run / "launch.json").stat().st_mtime
            latest = max((value for value in timestamps if value is not None), default=start)
            end = time_epoch(host.get("updated_at")) if host.get("phase") in TERMINAL_PHASES else None
            skill_count = meta.get("skill_count")
            if skill_count is None:
                skill_count = sum(1 for _ in (PROJECT / "skills" / meta["paper_id"] / "skill").rglob("SKILL.md"))
            self.snapshot = {
                "ok": True, "generated_at_bj": iso_time(time.time()), "runs": runs,
                "run": {"paper_id": meta["paper_id"], "run_dir": str(run),
                        "workspace": str(workspace), "pid": grader_pid,
                        "started_utc": iso_time(start, timezone.utc), "started_bj": iso_time(start),
                        "model": meta.get("model"), "host_phase": host.get("phase"),
                        "elapsed_seconds": max(0, int((end or time.time()) - start)),
                        "running": running, "last_event_utc": iso_time(latest, timezone.utc),
                        "last_event_bj": iso_time(latest), "phase": phase,
                        "error_count": sum(bool(attempt.get("error")) for attempt in checker["attempts"]),
                        "markers": {"complete": host.get("phase") == "complete",
                                    "error": host.get("phase") in TERMINAL_PHASES - {"complete", "smoke_complete"},
                                    "submission": (output_dir / "frozen/submission").is_dir()},
                        "skill_count": skill_count},
                "tasks": tasks, "hidden_task_count": sum(item["hidden_from_agent"] for item in tasks),
                "protocol": stats, "main_trace": self._main_trace(main_events),
                "checker_trace": checker, "files": self._files(run, workspace),
                "evaluation": evaluation,
                "diagnostics": {"main_bad_lines": self._cache(output_dir / "main.jsonl", CODEX_EVENTS).bad_lines,
                                "checker_bad_lines": sum(cache.bad_lines for path, cache in self.caches.items()
                                                         if path.parent == output_dir / "checker")},
            }
            return copy.deepcopy(self.snapshot)

    def get_snapshot(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.snapshot)

    def refresh_loop(self) -> None:
        while not self.stop_event.wait(2.0):
            try:
                self.refresh()
            except Exception:
                LOGGER.exception("Monitor refresh failed")


class Handler(BaseHTTPRequestHandler):
    monitor: Monitor

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("%s %s", self.address_string(), format % args)

    def send_json(self, value: object, status: int = 200) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self.send_json({"ok": True, "service": "codex-paperbench-monitor",
                                "time_bj": iso_time(time.time())})
                return
            if parsed.path.startswith("/api/"):
                snapshot = self.monitor.get_snapshot()
                if parsed.path == "/api/runs":
                    self.send_json({"ok": True, "runs": snapshot.get("runs", [])})
                elif parsed.path == "/api/state":
                    self.send_json({key: value for key, value in snapshot.items()
                                    if key not in {"main_trace", "checker_trace", "files"}})
                elif parsed.path == "/api/files":
                    self.send_json({"ok": True, "run": snapshot.get("run"),
                                    "files": snapshot.get("files", [])})
                elif parsed.path == "/api/trace":
                    role = parse_qs(parsed.query).get("role", ["main"])[0]
                    if role == "main":
                        self.send_json({"ok": True, "role": role,
                                        "events": snapshot.get("main_trace", [])})
                    elif role == "checker":
                        self.send_json({"ok": True, "role": role,
                                        **snapshot.get("checker_trace", {})})
                    else:
                        self.send_json({"ok": False, "error": "role must be main or checker"}, 400)
                else:
                    self.send_json({"ok": False, "error": "not found"}, 404)
                return
            relative = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
            target = (STATIC / relative).resolve(strict=False)
            if not inside(target, STATIC) or not target.is_file():
                self.send_error(404)
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except BrokenPipeError:
            pass
        except Exception:
            LOGGER.exception("HTTP request failed")
            self.send_json({"ok": False, "error": "monitor internal error"}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WEB_PORT", "8768")))
    parser.add_argument("--runs-root", default=os.getenv("CODEX_RUNS_ROOT", str(PROJECT / "runs")))
    parser.add_argument("--run-dir", default=os.getenv("CODEX_RUN_DIR"))
    args = parser.parse_args()
    root = Path(args.runs_root)
    if not root.is_dir():
        raise SystemExit(f"Codex runs root does not exist: {root}")
    Handler.monitor = Monitor(root, Path(args.run_dir) if args.run_dir else None)
    Handler.monitor.refresh()
    threading.Thread(target=Handler.monitor.refresh_loop, daemon=True).start()
    LOGGER.info("Listening on http://%s:%s (runs=%s)", args.host, args.port, root)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
