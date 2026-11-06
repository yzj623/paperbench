"""Runtime-owned asynchronous jobs for the ResearchAgent.

This module deliberately lives beside the ResearchAgent protocol instead of in
``runtimes.host_uv``.  The PaperBench runtime remains unchanged; a research
agent gets a small, task-local process registry which launches an argv vector,
redirects output to a file, and returns immediately.

The registry is not a general process supervisor.  In particular it does not
schedule GPUs and it does not promise to re-attach to jobs after a runtime
restart.  It does, however, make the current runtime's jobs visible to the
submission gate and records enough process identity to avoid PID-reuse kills.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .state_machine import AgentProcess, ResearchToolError
from .storage import PlanStore, atomic_json_write

_VIRTUAL_ROOTS = ("/home", "/root", "/submission", "/tmp", "/output")
_PROTECTED_ENV = {
    "PAPERBENCH_PROCESS_TOKEN",
    "HOME",
    "PATH",
    "VIRTUAL_ENV",
    "PAPERBENCH_WORKSPACE",
    "PAPERBENCH_LANDLOCK_RO",
    "PAPERBENCH_LANDLOCK_WO",
    "PAPERBENCH_LANDLOCK_RW",
    "HF_HOME",
    "TMPDIR",
}
_SENSITIVE_KEY = re.compile(
    r"(?:api.?key|access.?key|secret|password|passwd|credential|authorization|bearer|token)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(
    r"(?:sk-[A-Za-z0-9][A-Za-z0-9_.-]*|Bearer\s+\S+|-----BEGIN[\s\S]*?-----END)",
    re.IGNORECASE,
)
_PROC_START_TICKS = 22
_MAX_LOG_BYTES = 128 * 1024
_MAX_LOG_CHARS = 64 * 1024


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _redact_text(value: str) -> str:
    """Remove common credential forms from data returned to the model/logs."""

    value = _SENSITIVE_VALUE.sub("[REDACTED]", value)
    return re.sub(
        r"(?i)(api.?key|access.?key|secret|password|passwd|credential|authorization|bearer|token)"
        r"(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2[REDACTED]",
        value,
    )


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def redact_for_log(value: Any) -> Any:
    """Public bounded logging helper used by the solver's tool-call logger."""

    return _redact(value)


def _proc_start_ticks(pid: int) -> int | None:
    """Read Linux ``/proc/<pid>/stat`` starttime, without trusting PID alone."""

    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, OSError, UnicodeError):
        return None
    # ``comm`` can contain spaces and ')' so split after its final ')'.
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 1 :].split()
    try:
        return int(fields[_PROC_START_TICKS - 3])
    except (IndexError, ValueError):
        return None


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _read_proc_token(pid: int) -> str | None:
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    prefix = b"PAPERBENCH_PROCESS_TOKEN="
    for item in environ:
        if item.startswith(prefix):
            return item[len(prefix) :].decode("utf-8", errors="replace")
    return None


def _safe_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class ManagedJobManager:
    """Manage ResearchAgent-started asynchronous subprocesses.

    ``workspace_root`` is the real task workspace path.  Legacy virtual
    PaperBench paths are accepted for resumability, but new prompts give the
    model the real absolute path and all values are confined to this root.
    """

    def __init__(
        self,
        *,
        store: PlanStore,
        computer: Any,
        workspace_root: str | Path,
    ) -> None:
        self.store = store
        self.computer = computer
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.jobs_path = self.store.run_dir / "jobs.json"
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._watchers: dict[str, asyncio.Task[None]] = {}
        self._kill_requested: set[str] = set()
        self._closed = False
        self._lock = asyncio.Lock()
        self._load_jobs()

    @property
    def process_token(self) -> str | None:
        token = getattr(self.computer, "process_token", None)
        return str(token) if token else None

    def _load_jobs(self) -> None:
        if not self.jobs_path.exists():
            return
        try:
            payload = json.loads(self.jobs_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.store.events.append("background_job_lost", reason="invalid_jobs_json")
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
            self.store.events.append("background_job_lost", reason="invalid_jobs_json")
            return
        changed = False
        for item in payload["jobs"]:
            if not isinstance(item, dict) or not isinstance(item.get("job_id"), str):
                continue
            item = dict(item)
            job_id = item["job_id"]
            self._jobs[job_id] = item
            # This registry intentionally does not re-attach watchers across a
            # manager/runtime restart.  Keeping an old record as ``running``
            # would make its eventual exit look current even though no
            # Process handle can reap it reliably.
            if item.get("status") == "running":
                item["status"] = "lost"
                item["finished_at"] = _utc_now()
                item["loss_reason"] = "manager restarted; cross-runtime reattachment is unsupported"
                self.store.events.append(
                    "background_job_lost", job_id=job_id, reason=item["loss_reason"]
                )
                changed = True
        if changed:
            self._persist()

    def _persist(self) -> None:
        # Do not persist tokens or the expanded physical environment.
        jobs = [redact_for_log(item) for item in self._jobs.values()]
        atomic_json_write(self.jobs_path, {"version": 1, "jobs": jobs})

    def _virtual_to_physical(self, value: str, *, kind: str, for_write: bool = False) -> Path:
        if not isinstance(value, str) or not value or "\0" in value:
            raise ResearchToolError("INVALID_ARGUMENT", f"{kind} must be a non-empty path")
        raw = value.replace("\\", "/")
        parts = raw.split("/")
        if ".." in parts:
            raise ResearchToolError("PATH_ESCAPE", f"{kind} may not contain '..'", path=value)

        runtime_root = self.workspace_root.parent
        candidate = Path(raw)
        # Prompts use real absolute paths. Check those before interpreting a
        # path under /tmp as the legacy virtual /tmp root; otherwise a real
        # workspace such as /tmp/paperbench-... would be translated twice.
        try:
            physical = candidate.resolve(strict=False) if candidate.is_absolute() else None
        except OSError as exc:
            raise ResearchToolError("PATH_ESCAPE", f"cannot resolve {kind}: {value}") from exc
        lexical = Path(os.path.abspath(raw)) if candidate.is_absolute() else None
        if lexical is not None and _safe_within(lexical, runtime_root):
            if physical is None or not _safe_within(physical, runtime_root):
                raise ResearchToolError("PATH_ESCAPE", f"{kind} escapes the runtime", path=value)
            candidate = physical
        elif raw == "/home" or raw.startswith(("/home/", "/root/")) or raw == "/root":
            logical_root = "/home" if raw == "/home" or raw.startswith("/home/") else "/root"
            suffix = raw[len(logical_root) :].lstrip("/")
            candidate = self.workspace_root / suffix
        elif raw == "/submission" or raw.startswith("/submission/"):
            suffix = raw[len("/submission") :].lstrip("/")
            candidate = self.workspace_root / "submission" / suffix
        elif raw == "/tmp" or raw.startswith("/tmp/"):
            suffix = raw[len("/tmp") :].lstrip("/")
            candidate = runtime_root / "tmp" / suffix
        elif raw == "/output" or raw.startswith("/output/"):
            suffix = raw[len("/output") :].lstrip("/")
            candidate = runtime_root / "output" / suffix
        elif not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except OSError as exc:
            raise ResearchToolError("PATH_ESCAPE", f"cannot resolve {kind}: {value}") from exc
        allowed_roots = (self.workspace_root, runtime_root / "tmp", runtime_root / "output")
        if not any(_safe_within(resolved, root) for root in allowed_roots):
            raise ResearchToolError("PATH_ESCAPE", f"{kind} escapes the workspace", path=value)
        if for_write and _safe_within(resolved, self.workspace_root) and any(
            component.casefold() == "paper"
            for component in resolved.relative_to(self.workspace_root).parts
        ):
            raise ResearchToolError(
                "PROTECTED_PATH",
                f"{kind} may not write the public paper inputs",
                path=value,
            )
        return resolved

    def _translate_arg(self, value: str, *, executable: bool = False) -> str:
        if not isinstance(value, str) or "\0" in value:
            raise ResearchToolError("INVALID_ARGUMENT", "argv entries must be NUL-free strings")
        raw = value.replace("\\", "/")
        if ".." in raw.split("/"):
            raise ResearchToolError("PATH_ESCAPE", "argv paths may not contain '..'", argument=value)

        runtime_root = self.workspace_root.parent

        def physical_runtime_path(path: str, *, allow_external: bool = False) -> str | None:
            candidate = Path(path)
            if not candidate.is_absolute():
                return None
            try:
                lexical = Path(os.path.abspath(path))
                resolved = candidate.resolve(strict=False)
            except OSError as exc:
                raise ResearchToolError(
                    "PATH_ESCAPE", "cannot resolve argv path", argument=value
                ) from exc
            if not _safe_within(lexical, runtime_root):
                return None
            if not _safe_within(resolved, runtime_root):
                if allow_external:
                    # argv[0] may be an external interpreter; a symlink such
                    # as a uv venv python pointing at a managed interpreter
                    # is still a legitimate external executable.
                    return None
                raise ResearchToolError("PATH_ESCAPE", "argv path escapes the runtime", argument=value)
            return resolved.as_posix()

        def translate_path(path: str) -> str:
            physical = physical_runtime_path(path, allow_external=executable)
            if physical is not None:
                return physical
            if executable and Path(path).is_absolute():
                # argv[0] may name an external interpreter or executable
                # (e.g. a real /usr/bin or /tmp/... python).  Do not
                # reinterpret a real absolute path as a virtual root here;
                # the symlink-escape case was already rejected above by
                # ``physical_runtime_path``.
                return path
            if path == "/home" or path.startswith("/home/"):
                return self._virtual_to_physical(path, kind="argv").as_posix()
            if path == "/root" or path.startswith("/root/"):
                return self._virtual_to_physical(path, kind="argv").as_posix()
            if path == "/submission" or path.startswith("/submission/"):
                return self._virtual_to_physical(
                    "/home/submission" + path[len("/submission") :], kind="argv"
                ).as_posix()
            # /tmp and /output are task-local virtual roots and may be useful
            # for scratch outputs, but are still contained in this runtime.
            for root in ("/tmp", "/output"):
                if path == root or path.startswith(root + "/"):
                    suffix = path[len(root) :].lstrip("/")
                    candidate = self.workspace_root.parent / root.lstrip("/") / suffix
                    resolved = candidate.resolve(strict=False)
                    runtime_root = self.workspace_root.parent
                    if not _safe_within(resolved, runtime_root):
                        raise ResearchToolError("PATH_ESCAPE", "argv path escapes the runtime", argument=value)
                    return resolved.as_posix()
            if Path(path).is_absolute() and not executable:
                raise ResearchToolError(
                    "PATH_ESCAPE", "argv path escapes the workspace", argument=value
                )
            return path

        if (
            raw == "/home"
            or raw.startswith(("/home/", "/root/", "/submission/", "/tmp/", "/output/"))
            or raw in {"/root", "/submission", "/tmp", "/output"}
        ):
            return translate_path(raw)
        for marker in ("/home/", "/root/", "/submission/", "/tmp/", "/output/"):
            if marker in raw and raw.startswith("--") and "=" in raw:
                prefix, path = raw.split("=", 1)
                return prefix + "=" + translate_path(path)
        physical = physical_runtime_path(raw, allow_external=executable)
        if physical is not None:
            return physical
        if Path(raw).is_absolute() and not executable:
            raise ResearchToolError("PATH_ESCAPE", "argv path escapes the workspace", argument=value)
        return value

    def _base_environment(self) -> dict[str, str]:
        supplied = getattr(self.computer, "environment", None)
        if isinstance(supplied, Mapping):
            source = {str(key): str(value) for key, value in supplied.items()}
        else:
            source = {str(key): str(value) for key, value in os.environ.items()}
        # The model-driving API credential belongs to the solver, not to the
        # reproduced program.  Do not make credentials available to a job
        # whose raw output is intentionally stored in an Agent-readable file.
        return {
            key: value
            for key, value in source.items()
            if not _SENSITIVE_KEY.search(key) and not _SENSITIVE_VALUE.search(value)
        }

    def _build_environment(self, extra: Mapping[str, Any] | None) -> dict[str, str]:
        if extra is not None and not isinstance(extra, Mapping):
            raise ResearchToolError("INVALID_ARGUMENT", "env must be an object")
        environment = self._base_environment()
        for key, value in (extra or {}).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ResearchToolError("INVALID_ARGUMENT", "env keys and values must be strings")
            if "\0" in key or "\0" in value:
                raise ResearchToolError("INVALID_ARGUMENT", "environment keys and values must be NUL-free")
            if key.upper() in _PROTECTED_ENV or key.upper().startswith("LD_"):
                raise ResearchToolError(
                    "PROTECTED_ENV", f"env may not override runtime variable {key}", key=key
                )
            if _SENSITIVE_KEY.search(key) or _SENSITIVE_VALUE.search(value):
                raise ResearchToolError(
                    "SENSITIVE_ENV", "sensitive credentials may not be supplied through env", key=key
                )
            environment[key] = value
        token = self.process_token
        if token:
            environment["PAPERBENCH_PROCESS_TOKEN"] = token
        return environment

    def _identity_matches(self, job: Mapping[str, Any]) -> bool:
        pid = job.get("pid")
        start = job.get("proc_start_ticks")
        if not isinstance(pid, int) or not isinstance(start, int):
            return False
        if _proc_start_ticks(pid) != start:
            return False
        boot = job.get("boot_id")
        if boot and _boot_id() != boot:
            return False
        token = self.process_token
        if token and _read_proc_token(pid) != token:
            return False
        try:
            if isinstance(job.get("pgid"), int) and os.getpgid(pid) != job["pgid"]:
                return False
        except (ProcessLookupError, PermissionError, OSError):
            return False
        return True

    def _job_view(self, job: Mapping[str, Any]) -> dict[str, Any]:
        fields = (
            "job_id",
            "status",
            "pid",
            "pgid",
            "argv",
            "cwd",
            "log_path",
            "cuda_visible_devices",
            "created_at",
            "started_at",
            "finished_at",
            "exit_code",
            "loss_reason",
        )
        return {key: job[key] for key in fields if key in job}

    async def run_background(
        self,
        *,
        argv: Sequence[str],
        cwd: str = "/home/submission",
        log_path: str = "/home/logs/background.log",
        env: Mapping[str, Any] | None = None,
        env_extra: Mapping[str, Any] | None = None,
    ) -> str:
        """Start one detached argv process and return a structured job record."""

        if self._closed:
            raise ResearchToolError("RUNTIME_CLOSED", "background job manager is closed")
        if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)) or not argv:
            raise ResearchToolError("INVALID_ARGUMENT", "argv must be a non-empty array of strings")
        if any(not isinstance(item, str) for item in argv):
            raise ResearchToolError("INVALID_ARGUMENT", "argv must be a non-empty array of strings")
        if env is not None and env_extra is not None:
            raise ResearchToolError("INVALID_ARGUMENT", "provide env, not env_extra, not both")
        translated_argv = [
            self._translate_arg(item, executable=index == 0)
            for index, item in enumerate(argv)
        ]
        validator = getattr(self.computer, "validate_shell_command", None)
        executable_name = Path(translated_argv[0]).name
        if callable(validator) and executable_name in {
            "bash",
            "sh",
            "python",
            "python3",
        }:
            try:
                code_index = translated_argv.index("-c") + 1
            except (ValueError, IndexError):
                code_index = -1
            if 0 < code_index < len(translated_argv):
                try:
                    validator(translated_argv[code_index])
                except PermissionError as exc:
                    raise ResearchToolError(
                        "PATH_ESCAPE", str(exc), argument=translated_argv[code_index]
                    ) from exc
        if any(_SENSITIVE_VALUE.search(item) for item in translated_argv):
            raise ResearchToolError(
                "SENSITIVE_ARGUMENT", "credentials may not be supplied in background argv"
            )
        physical_cwd = self._virtual_to_physical(cwd, kind="cwd", for_write=True)
        if not physical_cwd.is_dir():
            raise ResearchToolError("INVALID_ARGUMENT", f"cwd is not an existing directory: {cwd}")
        physical_log = self._virtual_to_physical(log_path, kind="log_path", for_write=True)
        if physical_log.exists() and physical_log.is_dir():
            raise ResearchToolError("INVALID_ARGUMENT", "log_path must be a file")
        physical_log.parent.mkdir(parents=True, exist_ok=True)
        environment = self._build_environment(env if env is not None else env_extra)
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        try:
            log_handle = physical_log.open("ab")
        except OSError as exc:
            raise ResearchToolError("LOG_OPEN_FAILED", f"could not open log_path: {log_path}") from exc
        try:
            background_wrapper = getattr(self.computer, "wrap_background_argv", None)
            wrapper = getattr(self.computer, "wrap_subprocess_argv", None)
            runtime_name = str(
                getattr(self.computer, "environment", {}).get("PAPERBENCH_RUNTIME", "")
            )
            if runtime_name == "landlock-uv" and not callable(wrapper):
                raise ResearchToolError(
                    "SANDBOX_UNAVAILABLE",
                    "Landlock background jobs require the runtime launcher",
                )
            if callable(background_wrapper):
                launch_argv = background_wrapper(
                    list(translated_argv),
                    cwd=physical_cwd,
                    environment=environment,
                )
            else:
                launch_argv = (
                    wrapper(list(translated_argv))
                    if callable(wrapper)
                    else list(translated_argv)
                )
            process = await asyncio.create_subprocess_exec(
                *launch_argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log_handle,
                stderr=log_handle,
                cwd=physical_cwd,
                env=environment,
                start_new_session=True,
            )
        except OSError as exc:
            log_handle.close()
            raise ResearchToolError(
                "START_FAILED", "could not start the requested background executable"
            ) from exc
        except Exception:
            log_handle.close()
            raise
        finally:
            # The child retains its duplicated fd; the manager must not hold
            # the file open and thereby keep a log from reaching EOF.
            if not log_handle.closed:
                log_handle.close()
        start_ticks = _proc_start_ticks(process.pid)
        # A very short command may have exited before /proc can be sampled.
        # It is safe to retain such a completed record, but a still-live
        # process without an identity must never be admitted to the registry.
        if start_ticks is None:
            # Give the event loop one turn to deliver SIGCHLD for an
            # immediately-exiting command before deciding that identity
            # sampling failed.
            await asyncio.sleep(0)
            start_ticks = _proc_start_ticks(process.pid)
        if start_ticks is None and process.returncode is None:
            await self._stop_process_handle(process)
            raise ResearchToolError("PROCESS_IDENTITY_FAILED", "could not identify the new process")
        try:
            pgid = os.getpgid(process.pid)
        except (ProcessLookupError, PermissionError, OSError) as exc:
            if process.returncode is None:
                await self._stop_process_handle(process)
                raise ResearchToolError("PROCESS_IDENTITY_FAILED", "could not identify the new process group") from exc
            pgid = None
        record = {
            "job_id": job_id,
            "status": "running" if process.returncode is None else ("completed" if process.returncode == 0 else "failed"),
            "pid": process.pid,
            "pgid": pgid,
            "proc_start_ticks": start_ticks,
            "boot_id": _boot_id(),
            "argv": list(argv),
            "cwd": cwd,
            "log_path": log_path,
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
            "created_at": _utc_now(),
            "started_at": _utc_now(),
            "exit_code": process.returncode,
        }
        try:
            async with self._lock:
                self._jobs[job_id] = record
                self._processes[job_id] = process
                self._persist()
                self.store.events.append(
                    "background_job_started",
                    job_id=job_id,
                    pid=process.pid,
                    pgid=pgid,
                    argv=list(argv),
                    cwd=cwd,
                    log_path=log_path,
                    cuda_visible_devices=record["cuda_visible_devices"],
                )
                watcher = asyncio.create_task(
                    self._watch(job_id, process), name=f"paperbench-job-{job_id}"
                )
                self._watchers[job_id] = watcher
                watcher.add_done_callback(lambda _: self._watchers.pop(job_id, None))
        except BaseException:
            self._jobs.pop(job_id, None)
            self._processes.pop(job_id, None)
            try:
                self._persist()
            except Exception:
                pass
            await self._stop_verified_spawn(
                process=process, start_ticks=start_ticks, pgid=pgid
            )
            raise
        return json.dumps({"ok": True, **self._job_view(record)}, ensure_ascii=False)

    async def _watch(self, job_id: str, process: asyncio.subprocess.Process) -> None:
        try:
            exit_code = await process.wait()
        except asyncio.CancelledError:
            raise
        try:
            async with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    return
                killed = job_id in self._kill_requested
                already_finalized = record.get("finished_at") is not None
                record["status"] = (
                    "killed" if killed else ("completed" if exit_code == 0 else "failed")
                )
                record["exit_code"] = exit_code
                record["finished_at"] = _utc_now()
                self._persist()
                if not already_finalized:
                    self.store.events.append(
                        "background_job_killed" if killed else "background_job_exited",
                        job_id=job_id,
                        pid=record.get("pid"),
                        exit_code=exit_code,
                        status=record["status"],
                    )
                self._kill_requested.discard(job_id)
        finally:
            self._processes.pop(job_id, None)

    async def _refresh(self, job: dict[str, Any]) -> None:
        if job.get("status") != "running":
            return
        process = self._processes.get(str(job["job_id"]))
        if process is not None:
            # The in-memory handle is authoritative while it exists.  Once
            # the watcher observes the exit status it owns finalization; do
            # not scan /proc in that window because the PID may already be
            # gone or reused.
            if process.returncode is not None:
                return
            ticks = _proc_start_ticks(process.pid)
            if ticks is None:
                # The command exited so quickly that /proc no longer reports
                # it; the watcher will finalize the record.
                return
            if ticks != job.get("proc_start_ticks"):
                # The PID now names a different process: this handle is stale.
                self._mark_lost(job, "PID start-time changed; original process is gone")
            # start-time matches: the same live process (or its zombie) is
            # still handled by the watcher.  Never scan /proc environ here: a
            # zombie exposes an empty environ, which would falsely fail the
            # token check and mark a completed job as lost.
            return
        if not self._identity_matches(job):
            self._mark_lost(job, "PID/start-time/token identity check failed")

    def _mark_lost(self, job: dict[str, Any], reason: str) -> None:
        job_id = str(job["job_id"])
        killed = job_id in self._kill_requested
        job["status"] = "killed" if killed else "lost"
        job["finished_at"] = _utc_now()
        if not killed:
            job["loss_reason"] = reason
        self.store.events.append(
            "background_job_killed" if killed else "background_job_lost",
            job_id=job_id,
            reason=("job was terminated" if killed else reason),
        )
        self._persist()

    async def job_status(self, *, job_id: str | None = None) -> str:
        if job_id is not None and job_id not in self._jobs:
            raise ResearchToolError("UNKNOWN_JOB", f"unknown background job: {job_id}", job_id=job_id)
        async with self._lock:
            jobs = [self._jobs[job_id]] if job_id is not None else list(self._jobs.values())
            for job in jobs:
                await self._refresh(job)
            views = [self._job_view(job) for job in sorted(jobs, key=lambda item: item.get("created_at", ""))]
        return json.dumps({"ok": True, "jobs": views}, ensure_ascii=False)

    async def job_log(self, *, job_id: str, tail_lines: int = 40) -> str:
        if job_id not in self._jobs:
            raise ResearchToolError("UNKNOWN_JOB", f"unknown background job: {job_id}", job_id=job_id)
        if not isinstance(tail_lines, int) or isinstance(tail_lines, bool) or not 1 <= tail_lines <= 200:
            raise ResearchToolError("INVALID_ARGUMENT", "tail_lines must be between 1 and 200")
        job = self._jobs[job_id]
        try:
            path = self._virtual_to_physical(str(job["log_path"]), kind="log_path")
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - _MAX_LOG_BYTES))
                raw = stream.read(_MAX_LOG_BYTES)
        except OSError as exc:
            raise ResearchToolError("LOG_READ_FAILED", "could not read job log", job_id=job_id) from exc
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()[-tail_lines:]
        content = _redact_text("\n".join(lines))
        if len(content) > _MAX_LOG_CHARS:
            content = content[-_MAX_LOG_CHARS:]
        return json.dumps(
            {
                "ok": True,
                "job_id": job_id,
                "status": job.get("status"),
                "truncated": size > _MAX_LOG_BYTES,
                "lines": content,
            },
            ensure_ascii=False,
        )

    async def job_kill(self, *, job_id: str) -> str:
        if job_id not in self._jobs:
            raise ResearchToolError("UNKNOWN_JOB", f"unknown background job: {job_id}", job_id=job_id)
        async with self._lock:
            job = self._jobs[job_id]
            await self._refresh(job)
            if job.get("status") == "lost":
                raise ResearchToolError(
                    "STALE_JOB",
                    "job identity is stale; no signal was sent",
                    job_id=job_id,
                )
            if job.get("status") != "running":
                return json.dumps({"ok": True, **self._job_view(job)}, ensure_ascii=False)
            if not self._identity_matches(job):
                raise ResearchToolError("STALE_JOB", "job identity changed; no signal was sent", job_id=job_id)
            pgid = int(job["pgid"])
            self._kill_requested.add(job_id)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise ResearchToolError("KILL_FAILED", "permission denied while stopping this job", job_id=job_id) from exc
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not self._identity_matches(job):
                break
            await asyncio.sleep(0.05)
        if self._identity_matches(job):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                raise ResearchToolError("KILL_FAILED", "permission denied while force-stopping this job", job_id=job_id) from exc
        process = self._processes.get(job_id)
        if process is not None:
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
        async with self._lock:
            if job.get("status") == "running" and process is not None and process.returncode is not None:
                job["status"] = "killed"
                job["exit_code"] = process.returncode
                job["finished_at"] = _utc_now()
                self.store.events.append(
                    "background_job_killed",
                    job_id=job_id,
                    pid=job.get("pid"),
                    exit_code=process.returncode,
                    status="killed",
                )
            else:
                await self._refresh(job)
            self._kill_requested.discard(job_id)
            self._persist()
            return json.dumps({"ok": True, **self._job_view(job)}, ensure_ascii=False)

    async def active_processes(self) -> list[AgentProcess]:
        """Return active managed jobs after rechecking each process identity."""

        async with self._lock:
            for job in self._jobs.values():
                await self._refresh(job)
            self._persist()
            return [
                AgentProcess(int(job["pid"]), " ".join(str(item) for item in job.get("argv", [])))
                for job in self._jobs.values()
                if job.get("status") == "running" and self._identity_matches(job)
            ]

    async def close(self) -> None:
        """Stop watcher tasks without signalling jobs.

        HostUVRuntime remains the final owner of runtime cleanup and will stop
        token-matching descendants.  Cancelling watchers here avoids leaving
        asyncio tasks attached to an agent solver that is exiting, without
        risking a signal to a PID that may have been reused.
        """

        if self._closed:
            return
        self._closed = True
        current = asyncio.current_task()
        tasks = [task for task in self._watchers.values() if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._watchers.clear()

    async def _stop_verified_spawn(
        self,
        *,
        process: asyncio.subprocess.Process,
        start_ticks: int | None,
        pgid: int | None,
    ) -> None:
        """Best-effort cleanup after registration failure, without trusting PID alone."""

        if process.returncode is not None or start_ticks is None or pgid is None:
            return
        if _proc_start_ticks(process.pid) != start_ticks:
            return
        token = self.process_token
        if token and _read_proc_token(process.pid) != token:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            if _proc_start_ticks(process.pid) == start_ticks:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass

    @staticmethod
    async def _stop_process_handle(process: asyncio.subprocess.Process) -> None:
        """Clean up a just-spawned leader when /proc identity was unavailable.

        The asyncio process handle still owns the unreaped child, so signalling
        it cannot target a subsequently reused PID.  We intentionally do not
        guess a process group without verified identity.
        """

        if process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


# A descriptive alias for callers that want to emphasize the persistent
# registry rather than the process manager implementation.
ManagedJobRegistry = ManagedJobManager
