"""Standalone ResearchAgent rollout against an isolated PaperBench Docker task."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperbench.solvers.basicagent.completer import OpenAIResponsesTurnCompleterConfig

from .glm_completer import GLM47FlashTurnCompleterConfig
from .qwen37flash_completer import Qwen37FlashTurnCompleterConfig
from .deepseekv4flash_completer import DeepSeekV4FlashTurnCompleterConfig
from .ustc_completer import USTCDeepSeekFlashTurnCompleterConfig

from .solver import ResearchAgentSolver


@dataclass(frozen=True)
class ExecutionResult:
    output: bytes
    exit_code: int


@dataclass
class StandaloneTask:
    run_id: str
    run_group_id: str
    run_dir: str
    runs_dir: str
    agent_workspace: str = "/home"
    skipped_rollout: bool = False


class DockerComputer:
    """Small ComputerInterface-compatible adapter around one task container.

    The controller sees the host bind mount for safe read-only inspection. The
    agent's shell commands still execute inside the isolated Docker container.
    """

    def __init__(self, *, container: str, workspace: Path, command_timeout: int = 600):
        self.container = container
        self.workspace = workspace.resolve(strict=True)
        self.command_timeout = command_timeout
        self.process_token = f"research-agent-{uuid.uuid4().hex}"
        self.environment = {
            "PAPERBENCH_RUNTIME": "docker",
            "PAPERBENCH_PROCESS_TOKEN": self.process_token,
        }

    def resolve_path(self, path: str) -> str:
        raw = path.replace("\\", "/")
        if raw == "/home":
            return self.workspace.as_posix()
        if raw.startswith("/home/"):
            return (self.workspace / raw[len("/home/") :]).as_posix()
        return raw

    async def _run(self, argv: list[str], *, timeout: int | None = None) -> ExecutionResult:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(
                process.communicate(), timeout=timeout or self.command_timeout
            )
        except TimeoutError:
            process.send_signal(signal.SIGTERM)
            try:
                output, _ = await asyncio.wait_for(process.communicate(), timeout=10)
            except TimeoutError:
                process.kill()
                output, _ = await process.communicate()
            return ExecutionResult(
                output=output
                + f"\nERROR: command exceeded the {timeout or self.command_timeout} second tool timeout".encode(),
                exit_code=124,
            )
        return ExecutionResult(output=output, exit_code=int(process.returncode or 0))

    async def send_shell_command(self, cmd: str, *, idempotent: bool = False) -> ExecutionResult:
        del idempotent
        return await self._run(
            [
                "docker",
                "exec",
                "--workdir",
                "/home",
                "--env",
                f"PAPERBENCH_PROCESS_TOKEN={self.process_token}",
                self.container,
                "bash",
                "-lc",
                cmd,
            ]
        )

    async def check_shell_command(self, cmd: str, *, idempotent: bool = False) -> ExecutionResult:
        result = await self.send_shell_command(cmd, idempotent=idempotent)
        if result.exit_code != 0:
            raise RuntimeError(
                f"command failed with exit code {result.exit_code}:\n"
                f"{result.output.decode('utf-8', errors='replace')}"
            )
        return result

    async def upload(self, file: bytes, destination: str) -> None:
        physical = Path(self.resolve_path(destination))
        try:
            physical.relative_to(self.workspace)
        except ValueError as exc:
            raise PermissionError(f"upload destination is outside /home: {destination}") from exc
        physical.parent.mkdir(parents=True, exist_ok=True)
        physical.write_bytes(file)

    async def download(self, file: str) -> bytes:
        physical = Path(self.resolve_path(file))
        try:
            physical.relative_to(self.workspace)
        except ValueError:
            result = await self._run(
                ["docker", "exec", self.container, "bash", "-lc", f"cat -- {file!r}"],
                timeout=120,
            )
            if result.exit_code != 0:
                raise FileNotFoundError(file)
            return result.output
        return physical.read_bytes()

    async def active_processes(self) -> list[Any]:
        # Every shell launched by this adapter inherits a per-run token. A
        # detached child therefore remains detectable after its parent tool
        # call returns. The scanner excludes itself.
        script = (
            "import os; token=os.environ['PAPERBENCH_PROCESS_TOKEN'].encode(); me=os.getpid(); "
            "[(print(p,open('/proc/'+p+'/cmdline','rb').read().replace(b'\\0',b' ').decode('utf-8','replace'))) "
            "for p in os.listdir('/proc') if p.isdigit() and int(p)!=me and "
            "(lambda q: token in q)(open('/proc/'+p+'/environ','rb').read())]"
        )
        result = await self._run(
            [
                "docker",
                "exec",
                "--env",
                f"PAPERBENCH_PROCESS_TOKEN={self.process_token}",
                self.container,
                "python3",
                "-c",
                script,
            ],
            timeout=30,
        )
        if result.exit_code != 0:
            return []
        from .state_machine import AgentProcess

        processes: list[AgentProcess] = []
        for line in result.output.decode("utf-8", errors="replace").splitlines():
            pid, _, command = line.partition(" ")
            if pid.isdigit():
                processes.append(AgentProcess(pid=int(pid), command=command))
        return processes

    def wrap_background_argv(
        self, argv: list[str], *, cwd: Path, environment: dict[str, str]
    ) -> list[str]:
        wrapped = ["docker", "exec", "--workdir", cwd.as_posix()]
        for key, value in environment.items():
            wrapped.extend(["--env", f"{key}={value}"])
        wrapped.append(self.container)
        wrapped.extend(argv)
        return wrapped


async def run(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace).resolve(strict=True)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    task = StandaloneTask(
        run_id=args.run_id,
        run_group_id=args.run_group_id,
        run_dir=run_dir.as_posix(),
        runs_dir=run_dir.parent.as_posix(),
    )
    computer = DockerComputer(
        container=args.container,
        workspace=workspace,
        command_timeout=args.command_timeout,
    )
    if args.api_config == "openai-oauth":
        from .openai_oauth_completer import OpenAIOAuthTurnCompleterConfig, normalize_model
        oauth_options = {}
        if args.reasoning_effort is not None:
            oauth_options["reasoning"] = {"effort": args.reasoning_effort}
        completer_config = OpenAIOAuthTurnCompleterConfig(model=args.model or "gpt-6-luna", **oauth_options)
        (run_dir / "api-config.json").write_text(json.dumps({
            "provider": "openai-oauth", "model": normalize_model(completer_config.model),
            "reasoning": oauth_options.get("reasoning", "model default"),
            "context_window": completer_config.context_window,
        }, indent=2) + "\n", encoding="utf-8")
    elif args.api_config == "glm47flash":
        completer_config = GLM47FlashTurnCompleterConfig(model=args.model)
    elif args.api_config == "deepseekv4flash":
        completer_config = DeepSeekV4FlashTurnCompleterConfig(model=args.model)
    elif args.api_config == "qwen37flash":
        completer_config = Qwen37FlashTurnCompleterConfig(model=args.model)
    elif args.api_config == "ustc-deepseek-flash":
        completer_config = USTCDeepSeekFlashTurnCompleterConfig(model=args.model)
    else:
        completer_config = OpenAIResponsesTurnCompleterConfig(model=args.model)
    solver = ResearchAgentSolver(
        completer_config=completer_config,
        max_steps=None,
        planning_retries=None,
        plan_file=args.plan,
        checker_max_steps=args.checker_max_steps,
        max_review_rounds=args.max_review_rounds,
        mount_docker_socket=True,
    )
    started = time.time()
    try:
        output = await solver._run_agent(computer, task)
    except BaseException as exc:
        (run_dir / "rollout-error.json").write_text(
            json.dumps(
                {"type": type(exc).__name__, "message": str(exc), "time": time.time()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    else:
        (run_dir / "metadata.json").write_text(
            json.dumps(output.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "rollout-complete.json").write_text(
            json.dumps(
                {"run_id": args.run_id, "started_at": started, "finished_at": time.time()},
                indent=2,
            ),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-group-id", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--plan", default=None,
                        help="Use this frozen plan JSON, skipping the planner phase")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"),
                        default=os.getenv("RESEARCH_AGENT_REASONING_EFFORT"),
                        help="OpenAI OAuth reasoning effort (also RESEARCH_AGENT_REASONING_EFFORT)")
    parser.add_argument(
        "--api-config",
        choices=("openai", "openai-oauth", "glm47flash", "ustc-deepseek-flash", "qwen37flash", "deepseekv4flash"),
        default="openai",
    )
    parser.add_argument("--command-timeout", type=int, default=600)
    parser.add_argument("--checker-max-steps", type=int, default=300)
    parser.add_argument("--max-review-rounds", type=int, default=2,
                        help="Maximum number of full-submission checker rounds; zero keeps NoChecker only")
    args = parser.parse_args()
    if args.model is None:
        args.model = "gpt-6-luna" if args.api_config == "openai-oauth" else "gpt-4.1-mini"
    return args


def main() -> None:
    if os.name != "posix":
        raise SystemExit("Run this controller from WSL/Linux with Docker available.")
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
