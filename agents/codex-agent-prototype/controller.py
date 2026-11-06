#!/usr/bin/env python3
"""Standalone external PaperBench controller using local protocol and JSONL."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import signal
import shutil
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

from app_server import AppServer, AppServerError, server_command
from tool_schemas import role_tools, validate_call
from protocol.async_jobs import ManagedJobManager, redact_for_log
from protocol.checker import CheckAgentContext, parse_check_report
from protocol.plan import parse_plan, PlanValidationError, format_invalid_plan_error
from protocol.local_tools import ReadFileChunk, SearchFile, WorkspaceWriteFileTool
from protocol.prompts import (PLANNING_PROMPT, CHECKER_PROMPT, CHECK_INSTRUCTIONS,
                              continue_prompt, execution_prompt, render_plan_message)
from protocol.state_machine import AgentProcess, ResearchAgentTools, ResearchToolError, FinalSubmitResult
from protocol.storage import PlanStore, atomic_json_write


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_agent_view(store: PlanStore, output: Path, *, phase="execution"):
    atomic_text(output / "agent_view.json", render_plan_message(store.render())["content"])
    atomic_text(output / "phase.txt", phase + "\n")


def plan_prefix_messages(store: PlanStore):
    """Experiment-specific rendering belongs to the controller, not Codex."""
    return [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": render_plan_message(store.render())["content"]}]}]


class LocalRuntime:
    """Scan agent workloads while excluding Codex's own service processes."""
    def __init__(self, proc_root: Path = Path("/proc")):
        self.marker = secrets.token_hex(16)
        self.main_pid = None
        self.proc_root = proc_root

    def _is_code_mode_host(self, candidate: Path, parent_pid: int) -> bool:
        if self.main_pid is None or parent_pid != self.main_pid:
            return False
        try:
            server_exe = (self.proc_root / str(self.main_pid) / "exe").readlink()
            candidate_exe = (candidate / "exe").readlink()
        except OSError:
            return False
        return candidate_exe == server_exe.with_name("codex-code-mode-host")

    async def active_processes(self):
        markers = {f"{key}={self.marker}".encode() for key in
                   ("PI_RUN_AGENT_TOKEN", "PAPERBENCH_PROCESS_TOKEN")}
        active = []
        for candidate in self.proc_root.iterdir():
            if not candidate.name.isdigit() or int(candidate.name) == self.main_pid:
                continue
            try:
                if not markers.intersection((candidate / "environ").read_bytes().split(b"\0")):
                    continue
                stat = (candidate / "stat").read_text().rsplit(") ", 1)[1].split()
                if stat[0] == "Z":
                    continue
                if self._is_code_mode_host(candidate, int(stat[1])):
                    continue
                command = (candidate / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except (OSError, IndexError, ValueError):
                continue
            active.append(AgentProcess(pid=int(candidate.name), command=redact_for_log(command[:500])))
        return active

    async def close(self):
        for active in await self.active_processes():
            try:
                pgid = os.getpgid(active.pid)
                if pgid != os.getpgrp():
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    os.kill(active.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class LocalSnapshotter:
    def __init__(self, workspace, output):
        self.workspace, self.output = workspace, output

    async def freeze_submission(self):
        destination = self.output / "frozen"
        destination.mkdir()  # Refuse a second freeze, as in Pi.
        shutil.copytree(self.workspace / "submission", destination / "submission", symlinks=True)
        archive = self.output / "submissions/submission.tar.gz"
        archive.parent.mkdir(exist_ok=True)
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(destination / "submission", arcname="submission")
        return str(archive)


class LocalShellComputer:
    def __init__(self, workspace, runtime):
        self.workspace = workspace
        self.process_token = runtime.marker
        self.environment = {key: value for key, value in os.environ.items()
                            if not any(word in key.lower() for word in
                                       ("key", "secret", "token", "password", "credential", "authorization"))}
        self.environment["PI_RUN_AGENT_TOKEN"] = runtime.marker

    @staticmethod
    def wrap_background_argv(argv, *, cwd, environment):
        return list(argv)

    async def send_shell_command(self, cmd):
        process = await asyncio.create_subprocess_exec(
            "bash", "-lc", cmd, cwd=self.workspace, env=self.environment,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True)
        try:
            output, _ = await asyncio.wait_for(process.communicate(), 600)
        except (TimeoutError, asyncio.CancelledError):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            raise
        return SimpleNamespace(output=output)


def role_config(args, role):
    # Keep Codex's native router; dynamic tools extend it with our protocol.
    config = {
        "ppblab_dynamic_tools_only": False,
        "ppblab_unbounded_account_usage_retry": True,
    }
    if role == "main":
        # Match Pi's explicit paper-only skill root; retain native discovery,
        # without unrelated bundled system skills changing the experiment.
        config["skills"] = {"bundled": {"enabled": False}}
    else:
        config["skills"] = {
            "include_instructions": False, "bundled": {"enabled": False},
            "config": [{"path": str(path), "enabled": False}
                       for path in sorted(args.skills.rglob("SKILL.md"))]}
    if getattr(args, "strip_irrelevant_developer_context", False):
        # Config-only ablation: remove unrelated native developer context.
        # This also disables the matching capabilities (sub-agents, apps,
        # plugins and plugin skills), which is intentional for this variant.
        config.update({
            "agents": {"enabled": False},
            "include_collaboration_mode_instructions": False,
            "include_apps_instructions": False,
            "features": {"apps": False, "plugins": False, "tool_suggest": False},
        })
    return config


def role_environment(args, role, runtime=None):
    environment = os.environ.copy()
    home = args.codex_home / "role-homes" / role
    # Avoid implicit host-home auth, config, and personal skill discovery.
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    environment.update(CODEX_HOME=str(args.codex_home), HOME=str(home),
                       XDG_CONFIG_HOME=str(home / "config"))
    environment.pop("PI_RUN_AGENT_TOKEN", None)
    environment.pop("PAPERBENCH_PROCESS_TOKEN", None)
    if role == "main" and runtime is not None:
        # Native command descendants inherit this marker; LocalRuntime excludes
        # the app-server process itself when checking for lingering work.
        environment["PAPERBENCH_PROCESS_TOKEN"] = runtime.marker
    return environment, args.codex_home


async def run_role(args, *, role, system_prompt, user_prompt, tools, dispatch,
                   finished=None, runtime=None, trace_path=None, prefix_messages=None):
    """One role process, one persistent thread, sequential client tool execution."""
    timeout = getattr(args, f"{role}_timeout")
    environment, home = role_environment(args, role, runtime)
    async with asyncio.timeout(timeout):
        async with AppServer(cwd=args.workspace, codex_home=home, env=environment,
                             command=args.app_server, trace_path=trace_path or args.output / f"{role}.jsonl") as server:
            if runtime is not None:
                runtime.main_pid = server.process.pid
            model = await server.resolve_model(args.model)
            response = await server.request("thread/start", {
                "model": model, "cwd": str(args.workspace), "approvalPolicy": "never",
                # Main's client-executed tools can write and use the container's
                # network. Match those capabilities in Codex's permission context.
                "sandbox": "danger-full-access" if role == "main" else "read-only",
                "ephemeral": role != "main",
                "baseInstructions": system_prompt, "config": role_config(args, role),
                "dynamicTools": tools})
            thread_id = response["thread"]["id"]
            current_prefix = None

            async def refresh_prefix():
                nonlocal current_prefix
                if prefix_messages is None:
                    return
                messages = prefix_messages()
                if messages == current_prefix:
                    return
                # Await acknowledgement before turn/start or a tool response releases sampling.
                await server.set_prefix(thread_id, messages)
                current_prefix = json.loads(json.dumps(messages))
                atomic_json_write(args.output / "prefix_snapshots" / f"{secrets.token_hex(12)}.json",
                                  {"thread_id": thread_id, "messages": current_prefix})

            prompt = user_prompt
            while True:
                await refresh_prefix()
                started = await server.request("turn/start", {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": prompt}]})
                turn_id = started["turn"]["id"]
                final_text = ""
                while True:
                    event = await server.next_event()
                    method, params = event.get("method"), event.get("params", {})
                    if "id" in event:
                        if method != "item/tool/call":
                            if method in {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}:
                                await server.send({"id": event["id"], "result": {"decision": "decline"}})
                                continue
                            if method == "item/permissions/requestApproval":
                                await server.send({"id": event["id"], "result": {
                                    "permissions": {}, "scope": "turn"}})
                                continue
                            await server.send({"id": event["id"], "error": {
                                "code": -32601, "message": "Unsupported app-server request"}})
                            raise AppServerError("unexpected app-server request: " + str(method))
                        try:
                            if params.get("threadId") != thread_id or params.get("turnId") != turn_id or params.get("namespace"):
                                raise ValueError("tool call does not belong to the active role turn")
                            name, arguments = params["tool"], params["arguments"]
                            validate_call(tools, name, arguments)
                            value = await dispatch(name, arguments)
                            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                            success = True
                        except Exception as exc:
                            text = json.dumps({"ok": False, "error": {
                                "code": getattr(exc, "code", "INVALID_ARGUMENT"),
                                "message": redact_for_log(str(exc)),
                                "details": redact_for_log(getattr(exc, "details", {}))}})
                            success = False
                        # Transport/prefix errors fail closed, not as a recoverable tool error.
                        await refresh_prefix()
                        await server.respond(event["id"], text, success=success)
                        # Completion must follow the drained tool response, never precede it.
                        if finished is not None and finished():
                            return None
                    elif params.get("threadId") not in (None, thread_id):
                        continue
                    elif method == "error":
                        if not params.get("willRetry", False):
                            raise AppServerError("role provider error: " + str(redact_for_log(params.get("error", {}))))
                    elif method == "item/completed" and params.get("turnId") == turn_id:
                        item = params.get("item", {})
                        if item.get("type") == "agentMessage" and item.get("phase") in (None, "final_answer"):
                            final_text = item.get("text", "")
                    elif method == "turn/completed" and params.get("turn", {}).get("id") == turn_id:
                        turn = params["turn"]
                        if turn.get("status") != "completed" or turn.get("error"):
                            raise AppServerError("role turn failed: " + str(redact_for_log(turn.get("error") or turn.get("status"))))
                        break
                if role == "planner":
                    try:
                        return parse_plan(final_text)
                    except PlanValidationError as exc:
                        prompt = json.dumps(format_invalid_plan_error(exc), ensure_ascii=False)
                else:
                    prompt = (continue_prompt(checker_enabled=args.checker)
                              if role == "main"
                              else "Inspect the evidence and finish by calling submit_review.")


class CodexChecker:
    def __init__(self, args, serial):
        self.args, self.serial = args, serial

    async def review(self, context: CheckAgentContext):
        report = None

        async def dispatch(name, arguments):
            nonlocal report
            if name == "submit_review":
                report = parse_check_report(arguments).model_dump()
                return {"ok": True}
            raise ResearchToolError("UNKNOWN_TOOL", "checker only has submit_review as a custom tool")

        task = context.task.model_dump()
        visible_task = {**task, "status": "uncompleted"}
        prompt = (
            "Review this frozen subtask. Your only acceptance question is whether this "
            f"subtask was reproduced precisely; the briefing at {self.args.output / 'check_instruction.txt'}, "
            "the plan, and the statuses are context. Inspect the actual public paper "
            "and workspace with Codex's native read-only tools.\n\n"
            f"Subtask: {task}\nCurrent frozen subtask (including runtime-owned status):\n"
            f"{json.dumps([visible_task], ensure_ascii=False, indent=2)}\n"
            f"Primary-agent evidence pointer: {context.reproduction_description}")
        await run_role(self.args, role="checker", system_prompt=CHECKER_PROMPT.replace(
            "__WORKSPACE__", str(self.args.workspace)).replace(
                "/check_instruction.txt", str(self.args.output / "check_instruction.txt")),
            user_prompt=prompt, tools=role_tools("checker"), dispatch=dispatch,
            finished=lambda: report is not None,
            trace_path=self.args.output / "checker" / f"{task['id']}-review-{self.serial}.jsonl")
        atomic_json_write(self.args.output / "checker" / f"{task['id']}-review-{self.serial}.json", report)
        return report


class CodexCheckerFactory:
    def __init__(self, args):
        self.args, self.serial = args, 0

    async def build(self, *, completer_config, context):
        self.serial += 1
        return CodexChecker(self.args, self.serial)


class MainDispatcher:
    def __init__(self, args, manager, computer):
        self.args, self.manager, self.computer = args, manager, computer
        self.lock = asyncio.Lock()
        self.file_tools = ({
            "read_file_chunk": ReadFileChunk(),
            "search_file": SearchFile(),
            "write_file": WorkspaceWriteFileTool(args.workspace / "submission"),
        } if args.ra_file_tools else {})

    async def __call__(self, name, arguments):
        async with self.lock:
            self.manager._require_execution_phase()
            if name in {"submit_subtask", "skip_subtask", "final_submit"}:
                value = await getattr(self.manager, name)(**arguments)
            elif name in self.file_tools:
                return await self.file_tools[name].execute(self.computer, **arguments)
            else:
                raise ResearchToolError("UNKNOWN_TOOL", "unknown protocol tool")
            phase = "complete" if self.manager.state.phase == "evaluation" else "execution"
            atomic_json_write(self.args.output / "status.json", {
                "phase": phase, "accepted": self.manager.state.phase == "evaluation",
                "statuses": self.manager.store.statuses(),
                "failure_counts": self.manager.store.submission_failures()})
            publish_agent_view(self.manager.store, self.args.output, phase=phase)
            if isinstance(value, FinalSubmitResult):
                return {"accepted": value.accepted, "snapshot_path": value.snapshot_path, "errors": list(value.errors)}
            return value


def validate_inputs(args):
    for relative in ("paper/paper.md", "paper/addendum.md", "instructions.txt"):
        if not (args.workspace / relative).is_file():
            raise ValueError(f"missing public input: {args.workspace / relative}")
    if not (args.workspace / "submission").is_dir() or not any(args.skills.rglob("SKILL.md")):
        raise ValueError("submission directory or complete skills missing")


async def run(args):
    store = None
    runtime = LocalRuntime()
    jobs = None
    try:
        validate_inputs(args)
        if args.smoke_test:
            count = len(parse_plan(args.plan_input.read_text()).subtasks) if args.plan_input.is_file() else None
            print(json.dumps({"ok": True, "skills": len(list(args.skills.rglob("SKILL.md"))),
                              "fixed_plan_tasks": count, "command": server_command()}))
            return 0  # Never start a provider, inspect auth, or login.
        args.output.mkdir(parents=True, exist_ok=True)
        store = PlanStore(args.output / "protocol")
        os.chmod(store.run_dir, 0o700)
        atomic_json_write(args.output / "status.json", {"phase": "planning", "statuses": {}})
        if args.checker:
            atomic_text(args.output / "check_instruction.txt",
                        CHECK_INSTRUCTIONS.replace("__WORKSPACE__", str(args.workspace)))
        if args.plan_input.is_file():
            store.initialize(args.plan_input.read_text())
        else:
            async def dispatch(name, arguments):
                raise ResearchToolError("UNKNOWN_TOOL", "planner has no custom tools")

            plan = await run_role(args, role="planner", system_prompt=PLANNING_PROMPT.replace("__WORKSPACE__", str(args.workspace)),
                user_prompt=("The public task materials are mounted below. Read the complete files with "
                    "Codex's native read-only tools before planning:\n"
                    f"{args.workspace}/paper/paper.md\n{args.workspace}/paper/addendum.md\n"
                    f"{args.workspace}/paper/assets/\n{args.workspace}/instructions.txt\n"
                    f"Submit all implementation files under {args.workspace}/submission/. "
                    "The model/API is configured by the runtime; do not inspect or expose credentials."),
                tools=role_tools("planner"), dispatch=dispatch)
            store.initialize(plan)
        # Native discovery is retained only for main; source controls role tool specs.
        skills_link = args.workspace / ".agents/skills"
        skills_link.parent.mkdir(parents=True, exist_ok=True)
        if not skills_link.exists() and not skills_link.is_symlink():
            skills_link.symlink_to(args.skills, target_is_directory=True)
        elif skills_link.resolve() != args.skills.resolve():
            raise ValueError("existing .agents/skills does not point to the selected skills directory")
        computer = LocalShellComputer(args.workspace, runtime)
        jobs = ManagedJobManager(store=store, computer=computer, workspace_root=args.workspace)
        manager = ResearchAgentTools(
            store=store, runtime=runtime,
            checker_factory=CodexCheckerFactory(args) if args.checker else None,
            completer_config=None, public_paper_root=args.workspace / "paper",
            workspace_root=args.workspace,
            submission_root=args.workspace / "submission",
            instructions_file=args.workspace / "instructions.txt",
            checker_instructions_file=(args.output / "check_instruction.txt"
                                       if args.checker else None),
            checker_enabled=args.checker,
            snapshotter=LocalSnapshotter(args.workspace, args.output), job_manager=jobs)
        publish_agent_view(store, args.output)
        atomic_json_write(args.output / "status.json", {"phase": "execution", "statuses": store.statuses()})
        submit_hint = ("Call submit_subtask to request review of the current task."
                       if args.checker else
                       "Call submit_subtask when the current task is complete; it will be accepted directly.")
        main_prompt = execution_prompt(workspace=str(args.workspace), checker_enabled=args.checker)
        if args.ra_file_tools:
            main_prompt = main_prompt.replace(
                "Use Codex's native tools to inspect public inputs, edit source files, and run lightweight checks.",
                "Use read_file_chunk and search_file to inspect public inputs and source files; "
                "use write_file to create or completely replace source files. "
                "Use Codex's native command tools for lightweight checks."
            )
        await run_role(args, role="main",
            system_prompt=main_prompt,
            user_prompt=(f"Read {args.workspace}/instructions.txt and the public paper. The plan is frozen. "
                f"{submit_hint} "
                "Call final_submit only when all tasks are completed or validly skipped."),
            tools=role_tools("main", manager, include_ra_file_tools=args.ra_file_tools),
            dispatch=MainDispatcher(args, manager, computer),
            finished=lambda: manager.state.phase == "evaluation", runtime=runtime,
            prefix_messages=lambda: plan_prefix_messages(store))
        return 0
    except Exception as exc:
        error = redact_for_log(f"{type(exc).__name__}: {exc}")
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json_write(args.output / "status.json", {"phase": "failed", "error": error,
            "statuses": store.statuses() if store is not None and store.state_path.is_file() else {}})
        import sys
        print(error, file=sys.stderr)
        return 1
    finally:
        if jobs is not None:
            await jobs.close()
        await runtime.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--paper-id", required=True)
    result.add_argument("--workspace", type=Path, default=Path("/home"))
    result.add_argument("--output", "--output-dir", dest="output", type=Path, default=Path("/run-output"))
    result.add_argument("--skills", "--skills-root", dest="skills", type=Path, default=Path("/skills"))
    result.add_argument("--plan-input", "--plan", dest="plan_input", type=Path, default=Path("/run-input/plan.json"))
    result.add_argument("--codex-home", type=Path, default=Path(os.environ.get("CODEX_HOME", "/run-auth")))
    result.add_argument("--app-server", type=lambda value: [value], default=None,
                        help="standalone binary path; otherwise CODEX_APP_SERVER_COMMAND JSON argv or /opt/codex/bin/codex-app-server")
    result.add_argument("--model", default="gpt-6-luna")
    result.add_argument("--ra-file-tools", action="store_true",
                        help="Expose RA read_file_chunk, search_file, and write_file to the main agent")
    result.add_argument("--strip-irrelevant-developer-context", action=argparse.BooleanOptionalAction, default=False,
                        help="Disable native multi-agent, plugin/app and collaboration developer context")
    result.add_argument("--checker", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable the semantic Check Agent; --no-checker auto-accepts subtasks")
    result.add_argument("--planner-timeout", type=int, default=None)
    result.add_argument("--checker-timeout", type=int, default=None)
    result.add_argument("--main-timeout", type=int, default=None)
    result.add_argument("--smoke-test", action="store_true")
    return result


if __name__ == "__main__":
    cli = parser()
    args = cli.parse_args()
    if any(value is not None and value < 1 for value in (args.planner_timeout, args.checker_timeout, args.main_timeout)):
        cli.error("timeouts must be positive")
    for field in ("workspace", "output", "skills", "codex_home", "plan_input"):
        setattr(args, field, getattr(args, field).absolute())
    raise SystemExit(asyncio.run(run(args)))
