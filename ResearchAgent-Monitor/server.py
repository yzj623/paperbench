#!/usr/bin/env python3
"""Read-only PaperBench ResearchAgent monitor (stdlib only)."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import logging
from logging.handlers import RotatingFileHandler
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from datetime import datetime, timezone, timedelta
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

MONITOR_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = MONITOR_ROOT / "static"
CACHE_ROOT = MONITOR_ROOT / "cache"
LOG_ROOT = MONITOR_ROOT / "logs"
BEIJING = timezone(timedelta(hours=8))
MAX_EVENTS = 800
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_PREVIEW = 1800
SECRET_RE = re.compile(r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*[^\s,}\]]+")


def setup_logging() -> logging.Logger:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("research-agent-monitor")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(LOG_ROOT / "monitor.log", maxBytes=2_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


LOGGER = setup_logging()


def redact(value: object, limit: int = MAX_PREVIEW) -> str:
    text = str(value).replace("\x00", "")
    text = SECRET_RE.sub(r"\1=[REDACTED]", text)
    return text if len(text) <= limit else text[:limit] + f"… ({len(text) - limit} chars hidden)"


def iso_utc(epoch: float | None) -> str | None:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat() if epoch else None


def iso_bj(epoch: float | None) -> str | None:
    return datetime.fromtimestamp(epoch, BEIJING).isoformat() if epoch else None


def parse_time(value: object) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def safe_json(path: Path, default: object) -> object:
    try:
        if not path.is_file():
            return copy.deepcopy(default)
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except Exception as exc:
        LOGGER.warning("Cannot parse %s: %s", path, exc)
        return copy.deepcopy(default)


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


class IncrementalJSONL:
    """Incrementally consumes append-only JSONL without trusting the final line."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.identity: tuple[int, int] | None = None
        self.partial = b""
        self.events: collections.deque[dict] = collections.deque(maxlen=MAX_EVENTS)
        self.bad_lines = 0

    def refresh(self, on_event=None) -> int:
        new_count = 0
        try:
            stat = self.path.stat()
            ident = (stat.st_dev, stat.st_ino)
            if self.identity != ident or stat.st_size < self.offset:
                self.offset, self.partial, self.bad_lines = 0, b"", 0
                self.events.clear()
            self.identity = ident
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                while True:
                    chunk = stream.read(64 * 1024)
                    if not chunk:
                        break
                    self.offset += len(chunk)
                    data = self.partial + chunk
                    lines = data.split(b"\n")
                    self.partial = lines.pop()
                    if len(self.partial) > MAX_LINE_BYTES:
                        self.partial = b""
                        self.bad_lines += 1
                    for raw in lines:
                        if not raw.strip() or len(raw) > MAX_LINE_BYTES:
                            continue
                        try:
                            item = json.loads(raw)
                            if isinstance(item, dict):
                                self.events.append(item)
                                new_count += 1
                                if on_event is not None:
                                    on_event(item)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            self.bad_lines += 1
        except FileNotFoundError:
            pass
        except Exception as exc:
            LOGGER.exception("JSONL refresh failed for %s: %s", self.path, exc)
        return new_count



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
    def __init__(self, experiments_root: Path, run_root: Path | None = None,
                 batch_summary: Path | None = None):
        self.root = experiments_root.resolve(strict=True)
        self.pinned_root = run_root.resolve(strict=True) if run_root else None
        self.batch_summary = batch_summary.resolve(strict=True) if batch_summary else None
        if self.pinned_root is not None and self.batch_summary is not None:
            raise ValueError("Choose either a pinned run or a serial batch")
        if self.batch_summary is not None and not inside(self.batch_summary, self.root):
            raise ValueError("Batch summary must be inside experiments root")
        if self.pinned_root is not None and not inside(self.pinned_root, self.root):
            raise ValueError("Pinned run must be inside experiments root")
        self.lock = threading.RLock()
        self.caches: dict[Path, IncrementalJSONL] = {}
        self.snapshot: dict = {}
        self.protocol = collections.Counter()
        self.protocol_results: dict[str, collections.deque[str]] = {
            "submit_subtask": collections.deque(),
            "skip_subtask": collections.deque(),
            "final_submit": collections.deque(),
        }
        self.active_root: Path | None = None
        self.stop_event = threading.Event()

    def _active_from_processes(self) -> tuple[Path | None, int | None, float | None]:
        proc = Path("/proc")
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                # Discover any ResearchAgent run.  The monitor is shared by
                # different papers and modes, so do not hard-code a paper
                # prefix such as ``all-in-one_`` here.  The run-dir boundary
                # check below remains the guard against unrelated processes.
                if "research_agent.runner" not in raw:
                    continue
                match = re.search(r"--run-dir\s+(\S+)", raw)
                if not match:
                    continue
                run_dir = Path(match.group(1)).resolve(strict=True)
                run_root = run_dir.parent
                if not inside(run_root, self.root) or run_dir.name != "agent-run":
                    continue
                if self.pinned_root is not None and run_root != self.pinned_root:
                    continue
                if self.batch_summary is not None and run_root.parent != self.batch_summary.parent:
                    continue
                stat = entry.stat()
                ticks = int((entry / "stat").read_text().split()[21])
                boot = float(Path("/proc/stat").read_text().split("btime ", 1)[1].splitlines()[0])
                started = boot + ticks / os.sysconf("SC_CLK_TCK")
                return run_root, int(entry.name), started
            except (OSError, ValueError, IndexError):
                continue
        return None, None, None

    def _valid_runs(self, active: Path | None) -> list[dict]:
        found = []
        if self.batch_summary is not None:
            batch = safe_json(self.batch_summary, {})
            if not isinstance(batch, dict):
                return found
            records = batch.get("experiments", [])
            if not isinstance(records, list):
                return found
            for record in records:
                if not isinstance(record, dict) or not isinstance(record.get("run_dir"), str):
                    continue
                run = Path(record["run_dir"]).resolve(strict=False)
                if run.parent != self.batch_summary.parent or not run.is_dir():
                    continue
                found.append({"paper_id": batch.get("paper_id", ""), "run_dir": str(run),
                              "active": active == run, "config": record.get("config"),
                              "batch_status": record.get("status"),
                              "modified_bj": iso_bj(run.stat().st_mtime)})
            # The controller appends a record when it actually starts a stage.
            # Keep its order, rather than interpreting filesystem mtimes as progress.
            return found
        for paper_dir in self.root.iterdir():
            if not paper_dir.is_dir():
                continue
            for run in paper_dir.iterdir():
                if not run.is_dir() or not inside(run, self.root):
                    continue
                if self.pinned_root is not None and run.resolve() != self.pinned_root:
                    continue
                state = run / "agent-run" / "research_agent"
                if not (state / "plan.json").is_file() or not (run / "workspace" / "submission").is_dir():
                    continue
                found.append({
                    "paper_id": paper_dir.name,
                    "run_dir": str(run),
                    "active": active == run.resolve(strict=False),
                    "modified_bj": iso_bj(run.stat().st_mtime),
                })
        found.sort(key=lambda x: (not x["active"], -Path(x["run_dir"]).stat().st_mtime))
        return found

    def _cache(self, path: Path) -> IncrementalJSONL:
        cache = self.caches.get(path)
        if cache is None:
            cache = self.caches[path] = IncrementalJSONL(path)
        return cache

    def _reset_for_run(self, run: Path) -> None:
        if self.active_root == run:
            return
        self.active_root = run
        self.caches.clear()
        self.protocol.clear()
        for queue in self.protocol_results.values():
            queue.clear()

    def _protocol_event(self, event: dict) -> None:
        names = self.protocol_results
        kind, tool = event.get("event"), event.get("tool")
        if tool not in names:
            return
        if kind == "agent_tool_call":
            self.protocol[f"{tool}_calls"] += 1
            names[tool].append(tool)
        elif kind == "agent_tool_result" and names[tool]:
            names[tool].popleft()
            raw = str(event.get("result", ""))
            try:
                result = json.loads(raw)
            except Exception:
                result = {}
            if tool == "submit_subtask":
                if "changed to completed" in raw:
                    self.protocol["submit_passed"] += 1
                elif "changed to uncompleted" in raw or result.get("ok") is False:
                    self.protocol["submit_rejected"] += 1
            elif tool == "final_submit" and result.get("accepted"):
                self.protocol["final_accepted"] += 1

    def _protocol_stats(self) -> dict:
        return {
            "submit_task_calls": self.protocol["submit_subtask_calls"],
            "submit_passed": self.protocol["submit_passed"],
            "submit_rejected": self.protocol["submit_rejected"],
            "skip_task_calls": self.protocol["skip_subtask_calls"],
            "final_submit_calls": self.protocol["final_submit_calls"],
            "final_accepted": self.protocol["final_accepted"],
        }

    def _normalize_tool(self, event: dict) -> dict:
        args = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
        paths = []
        for key in ("path", "cwd", "log_path"):
            if key in args:
                paths.append(redact(args[key], 400))
        summary = ""
        if event.get("event", "").endswith("call"):
            if event.get("tool") == "write_file":
                summary = f"write {redact(args.get('path', ''), 500)}"
            elif event.get("tool") == "bash":
                summary = redact(args.get("cmd", args.get("command", "")), 600)
            else:
                summary = redact({k: v for k, v in args.items() if k != "content"}, 700)
        else:
            summary = redact(event.get("result", event.get("error", "")), 900)
        return {
            "timestamp": event.get("timestamp"),
            "time_bj": iso_bj(parse_time(event.get("timestamp"))),
            "event": event.get("event", "unknown"),
            "tool": event.get("tool"),
            "task_id": event.get("task_id") or args.get("subtask_id"),
            "paths": paths,
            "summary": summary,
        }

    def _checker_trace(self, events: list[dict], tasks: dict[str, dict], main: list[dict]) -> dict:
        submissions = {}
        for e in main:
            if e.get("event") == "agent_tool_call" and e.get("tool") == "submit_subtask":
                args = e.get("arguments", {})
                submissions[args.get("subtask_id")] = redact(args.get("reproduction_description", ""), 1600)
        attempts, current = [], None
        for e in events:
            kind = e.get("event")
            if kind == "checker_started":
                tid = e.get("task_id")
                current = {
                    "attempt": e.get("attempt"), "task_id": tid,
                    "task_name": tasks.get(tid, {}).get("name"),
                    "reproduction_description": submissions.get(tid, ""),
                    "started": e.get("timestamp"), "started_bj": iso_bj(parse_time(e.get("timestamp"))),
                    "reads": [], "events": [], "result": None, "reason": None,
                    "issues": [], "format_error": False, "error": None, "duration_seconds": None,
                }
                attempts.append(current)
            if current is None:
                continue
            current["events"].append(self._normalize_tool(e))
            if kind == "checker_tool_call" and e.get("tool") in {"read_file", "list_directory", "search_file"}:
                path = (e.get("arguments") or {}).get("path")
                if path:
                    current["reads"].append(redact(path, 600))
            if kind == "checker_response":
                report = e.get("report") if isinstance(e.get("report"), dict) else {}
                current["result"] = "accepted" if report.get("accepted") else "rejected"
                current["reason"] = redact(report.get("summary", ""), 1800)
                current["issues"] = report.get("issues", [])[:30]
            elif kind == "checker_response_invalid":
                current["format_error"] = True
                current["error"] = redact(e.get("details", ""), 1200)
            elif kind in {"checker_call_error", "checker_tool_error"}:
                current["error"] = redact(e.get("error", ""), 1200)
            if kind in {"checker_response", "checker_response_invalid", "checker_call_error"}:
                end = parse_time(e.get("timestamp")); start = parse_time(current.get("started"))
                if end and start:
                    current["duration_seconds"] = round(end - start, 3)
        return {"message": None if attempts else "尚未调用 Checker", "attempts": attempts[-80:]}

    def _files(self, run: Path) -> list[dict]:
        roots = [(run / "workspace" / "submission", "workspace/submission")]
        for rel in ("evaluation", "export/submission", "agent-run"):
            candidate = run / rel
            if candidate.is_dir():
                roots.append((candidate, rel))
        allowed_reports = {"metadata.json", "rollout-error.json", "rollout-complete.json", "status.json", "results.json", "grading.json", "report.json"}
        items = []
        for base, prefix in roots:
            for directory, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "node_modules"}]
                for name in files:
                    p = Path(directory) / name
                    rel = f"{prefix}/{p.relative_to(base).as_posix()}"
                    if prefix == "agent-run" and name not in allowed_reports:
                        continue
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    items.append({"path": rel, "name": name, "size": st.st_size,
                                  "modified_bj": iso_bj(st.st_mtime), "modified_utc": iso_utc(st.st_mtime)})
                    if len(items) >= 3000:
                        return items
        return sorted(items, key=lambda x: x["path"])

    def _agent_actions(self, path: Path) -> list[dict]:
        """Extract recent assistant prose from a bounded tail of the rendered log."""
        try:
            with path.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 128 * 1024))
                text = stream.read().decode("utf-8", errors="replace")
        except (FileNotFoundError, OSError):
            return []
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
        blocks = []
        current = None
        for line in text.splitlines():
            if "─ assistant " in line:
                current = []
                continue
            if current is not None and line.startswith("╰"):
                content = "\n".join(current).strip()
                if content:
                    blocks.append(content)
                current = None
                continue
            if current is not None:
                cleaned = line.strip().strip("│").strip()
                if cleaned:
                    current.append(cleaned)
        mtime = path.stat().st_mtime if path.exists() else None
        return [{"timestamp": iso_utc(mtime), "time_bj": iso_bj(mtime),
                 "event": "assistant_summary", "tool": None, "task_id": None,
                 "paths": [], "summary": redact(block, 1200)} for block in blocks[-12:]]

    def refresh(self) -> dict:
        with self.lock:
            active, pid, process_start = self._active_from_processes()
            runs = self._valid_runs(active)
            run = active
            if self.batch_summary is not None and run is None and runs:
                run = Path(runs[-1]["run_dir"])
            if run is None and self.active_root is not None and any(
                    Path(item["run_dir"]) == self.active_root for item in runs):
                run = self.active_root
            if run is None and runs:
                run = Path(runs[0]["run_dir"]).resolve(strict=True)
            if run is None or not inside(run, self.root):
                self.snapshot = {"ok": True, "message": "没有有效的 ResearchAgent 实例", "runs": runs}
                return copy.deepcopy(self.snapshot)
            self._reset_for_run(run)
            run_info = next((item for item in runs if Path(item["run_dir"]) == run), {})
            state_dir = run / "agent-run" / "research_agent"
            plan_obj = safe_json(state_dir / "plan.json", {"subtasks": []})
            state_obj = safe_json(state_dir / "plan_state.json", {"statuses": {}})
            skill_obj = safe_json(state_dir / "skill_selection.json", {"task_selections": {}})
            tasks_list = plan_obj.get("subtasks", []) if isinstance(plan_obj, dict) else []
            statuses = state_obj.get("statuses", {}) if isinstance(state_obj, dict) else {}
            skill_selections = (
                skill_obj.get("task_selections", {})
                if isinstance(skill_obj, dict)
                else {}
            )
            task_map = {t.get("id"): t for t in tasks_list if isinstance(t, dict)}
            tool_cache = self._cache(state_dir / "tool_calls.jsonl")
            checker_cache = self._cache(state_dir / "checker_calls.jsonl")
            event_cache = self._cache(state_dir / "events.jsonl")
            tool_cache.refresh(self._protocol_event)
            checker_cache.refresh()
            event_cache.refresh()
            protocol = self._protocol_stats()
            checker = self._checker_trace(list(checker_cache.events), task_map, list(tool_cache.events))
            checker_by_task = {}
            for attempt in checker["attempts"]:
                checker_by_task[attempt.get("task_id")] = attempt
            calls_by_task = collections.Counter()
            for event in tool_cache.events:
                if event.get("event") == "agent_tool_call" and event.get("tool") == "submit_subtask":
                    calls_by_task[(event.get("arguments") or {}).get("subtask_id")] += 1
            current_index = next(
                (
                    index
                    for index, task in enumerate(tasks_list)
                    if statuses.get(task.get("id"), "uncompleted") == "uncompleted"
                ),
                len(tasks_list),
            )
            tasks = []
            for index, task in enumerate(tasks_list):
                tid = task.get("id")
                latest = checker_by_task.get(tid, {})
                selected_skills = skill_selections.get(tid, [])
                skill_ids = [
                    item.get("skill_id")
                    for item in selected_skills
                    if isinstance(item, dict) and item.get("skill_id")
                ]
                tasks.append({
                    "task_id": tid, "name": task.get("name"), "description": task.get("description"),
                    "skill_ids": skill_ids,
                    "status": statuses.get(tid, "uncompleted"), "submit_attempts": calls_by_task[tid],
                    "checker_result": latest.get("result"), "checker_reason": latest.get("reason"),
                    "hidden_from_agent": index > current_index,
                })
            timestamps = []
            for cache in (tool_cache, checker_cache, event_cache):
                timestamps.extend(filter(None, (parse_time(e.get("timestamp")) for e in cache.events)))
            markers = {
                name: (run / rel).is_file() for name, rel in {
                    "complete": "agent-run/rollout-complete.json", "error": "agent-run/rollout-error.json",
                    "submission": "FINAL_SUBMISSION_PATH",
                }.items()
            }
            running = active == run
            grader_pid, _ = find_grading_process(run)
            running = running or grader_pid is not None
            evaluation = read_evaluation(run, grading_active=grader_pid is not None)
            if grader_pid is not None:
                phase = "评分中"
            elif evaluation["status"] == "complete":
                phase = "评分完成"
            elif markers["error"]:
                phase = "实验失败"
            elif markers["complete"] and (run / "evaluation").is_dir():
                phase = "评分阶段或已完成"
            elif markers["submission"]:
                phase = "评分阶段"
            elif checker["attempts"]:
                phase = "Checker 验收"
            elif (state_dir / "plan.json").is_file():
                phase = "Main Agent 实现"
            else:
                phase = "规划"
            if self.batch_summary is not None:
                batch_status = run_info.get("batch_status")
                if batch_status == "failed":
                    phase = "实验或评分失败"
                elif batch_status == "interrupted":
                    phase = "实验已中断"
                elif batch_status == "grading" and evaluation["status"] != "complete":
                    phase = "评分中"
            errors = sum(1 for e in list(tool_cache.events) + list(checker_cache.events)
                         if "error" in str(e.get("event", "")).lower() or e.get("error"))
            start = process_start or run.stat().st_ctime
            latest = max(timestamps, default=run.stat().st_mtime)
            main_trace = [self._normalize_tool(e) for e in list(tool_cache.events)[-400:]]
            main_trace.extend(self._agent_actions(run / "agent-run" / "agent.log"))
            self.snapshot = {
                "ok": True, "generated_at_bj": iso_bj(time.time()), "runs": runs,
                "run": {"paper_id": run_info.get("paper_id", run.parent.name),
                        "config": run_info.get("config"), "run_dir": str(run), "pid": pid,
                        "started_utc": iso_utc(start), "started_bj": iso_bj(start),
                        "elapsed_seconds": max(0, int(time.time() - start)), "running": running,
                        "last_event_utc": iso_utc(latest), "last_event_bj": iso_bj(latest),
                        "phase": phase, "error_count": errors, "markers": markers},
                "tasks": tasks, "hidden_task_count": sum(t["hidden_from_agent"] for t in tasks),
                "protocol": protocol,
                "main_trace": main_trace,
                "checker_trace": checker, "files": self._files(run),
                "evaluation": evaluation,
                "diagnostics": {"tool_bad_lines": tool_cache.bad_lines,
                                "checker_bad_lines": checker_cache.bad_lines,
                                "events_bad_lines": event_cache.bad_lines},
            }
            return copy.deepcopy(self.snapshot)

    def get_snapshot(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.snapshot)

    def refresh_loop(self) -> None:
        while not self.stop_event.wait(2.0):
            try:
                self.refresh()
            except Exception as exc:
                LOGGER.exception("Background refresh failed: %s", exc)


class Handler(BaseHTTPRequestHandler):
    monitor: Monitor

    def log_message(self, fmt: str, *args: object) -> None:
        LOGGER.info("%s %s", self.address_string(), fmt % args)

    def send_json(self, data: object, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(raw)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self.send_json({"ok": True, "service": "research-agent-monitor", "time_bj": iso_bj(time.time())})
                return
            if parsed.path.startswith("/api/"):
                snap = self.monitor.get_snapshot()
                if parsed.path == "/api/runs": self.send_json({"ok": True, "runs": snap.get("runs", [])})
                elif parsed.path == "/api/state": self.send_json({k: v for k, v in snap.items() if k not in {"main_trace", "checker_trace", "files"}})
                elif parsed.path == "/api/files": self.send_json({"ok": True, "run": snap.get("run"), "files": snap.get("files", [])})
                elif parsed.path == "/api/trace":
                    role = parse_qs(parsed.query).get("role", ["main"])[0]
                    if role == "main": self.send_json({"ok": True, "role": role, "events": snap.get("main_trace", [])})
                    elif role == "checker": self.send_json({"ok": True, "role": role, **snap.get("checker_trace", {})})
                    else: self.send_json({"ok": False, "error": "role must be main or checker"}, 400)
                else: self.send_json({"ok": False, "error": "not found"}, 404)
                return
            rel = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
            target = (STATIC_ROOT / rel).resolve(strict=False)
            if not inside(target, STATIC_ROOT) or not target.is_file():
                self.send_error(404); return
            raw = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(raw)
        except BrokenPipeError:
            pass
        except Exception as exc:
            LOGGER.exception("Request failed: %s", exc)
            self.send_json({"ok": False, "error": "monitor internal error"}, 500)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("WEB_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WEB_PORT", "8765")))
    parser.add_argument("--experiments-root", default=os.getenv("EXPERIMENTS_ROOT", str(Path(__file__).resolve().parent.parent / "experiments")))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run-dir", default=os.getenv("RA_RUN_DIR"), help="Pin this experiment, including after grading finishes")
    selection.add_argument("--batch-summary", default=os.getenv("RA_BATCH_SUMMARY"),
                           help="Follow the current experiment in a serial batch summary.json")
    args = parser.parse_args()
    root = Path(args.experiments_root)
    if not root.is_dir():
        raise SystemExit(f"Experiments root does not exist: {root}")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    Handler.monitor = Monitor(root, Path(args.run_dir) if args.run_dir else None,
                              Path(args.batch_summary) if args.batch_summary else None)
    Handler.monitor.refresh()
    threading.Thread(target=Handler.monitor.refresh_loop, name="snapshot-refresh", daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOGGER.info("Listening on http://%s:%s with root %s", args.host, args.port, root.resolve())
    server.serve_forever()


if __name__ == "__main__":
    main()
