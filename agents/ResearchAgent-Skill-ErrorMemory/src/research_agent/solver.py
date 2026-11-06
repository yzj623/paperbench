"""Runnable PaperBench solver for the planner/checker research-agent protocol."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import blobfile as bf
import tiktoken
from openai import LengthFinishReasonError
from typing_extensions import override

import chz
from paperbench.solvers.base import BasePBSolver

from .audit import audited_request
from .api_settings import CompatibleAPISettings
from .async_jobs import ManagedJobManager, redact_for_log
from .context import MessageWindow
from .prompts import (
    CHECK_INSTRUCTIONS,
    PLANNING_PROMPT,
    execution_prompt_for_skills,
    render_plan_message,
    render_workspace_prompt,
)
from .skills import SkillCatalog, SkillCoordinator, SkillSelector
from .session import ResearchAgentSession
from .state_machine import AgentProcess, ResearchAgentTools, Snapshotter
from .storage import PlanStore

_WORKSPACE_BASE = "/home"
_CONTEXT_TRIM_TRIGGER_RATIO = 0.60
_CONTEXT_TRIM_TARGET_RATIO = 0.45
# Consecutive identical tool calls before the agent is warned to change strategy.
_REPETITION_WARNING_THRESHOLD = 5
# Message appended after duplicate tool-call units are dropped on a 400 rejection.
_REPETITION_RECOVERY_MESSAGE = (
    "The conversation history contained repeated identical tool calls, which the provider "
    "rejected. Those duplicate tool calls have now been removed from the history. Continue "
    "the current subtask, and do not issue an identical tool call again: change the command or "
    "arguments, or accept the result you already have and move on."
)


def _log_value(value: Any, *, limit: int = 8_000) -> str:
    """Bound tool output copied into the persistent JSONL tool log."""

    value = redact_for_log(value)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated {len(text) - limit} chars]"


def _estimate_request_tokens(*, completer: Any, messages: Sequence[dict[str, Any]]) -> int:
    """Estimate the complete compatible-chat request with the configured tokenizer."""

    encoding_name = str(getattr(completer, "encoding_name", "o200k_base"))
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except ValueError:
        encoding = tiktoken.get_encoding("o200k_base")
    payload: dict[str, Any] = {"messages": list(messages)}
    configured_tools = getattr(completer, "tools", None)
    if isinstance(configured_tools, list):
        payload["tools"] = configured_tools
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    return len(encoding.encode_ordinary(serialized))


def _trim_execution_context(
    *,
    session: ResearchAgentSession,
    completer: Any,
    context_window: int,
    force: bool = False,
) -> tuple[int, int, int]:
    """At 60% usage, remove oldest complete units until usage is at most 45%."""

    before = _estimate_request_tokens(
        completer=completer,
        messages=session.execution_messages(),
    )
    trigger = max(1, int(context_window * _CONTEXT_TRIM_TRIGGER_RATIO))
    target = max(1, int(context_window * _CONTEXT_TRIM_TARGET_RATIO))
    if not force and before < trigger:
        return before, before, 0

    removed = 0
    current = before
    # A provider can still report an overflow when its tokenizer/accounting
    # differs from our compatible-model estimate. In that case force at least
    # one complete-unit removal, then continue toward the normal 45% target.
    if force and current <= target and session.window is not None:
        if session.window.trim_oldest_unit():
            removed += 1
            current = _estimate_request_tokens(
                completer=completer,
                messages=session.execution_messages(),
            )
    while current > target and session.window is not None:
        if not session.window.trim_oldest_unit():
            break
        removed += 1
        current = _estimate_request_tokens(
            completer=completer,
            messages=session.execution_messages(),
        )
    return before, current, removed


def _default_computer_runtime() -> Any:
    from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerRuntime

    return AlcatrazComputerRuntime()


def _default_compatible_config() -> Any:
    from paperbench.solvers.basicagent.completer import OpenAIResponsesTurnCompleterConfig

    return OpenAIResponsesTurnCompleterConfig(model="gpt-4.1-mini")


@chz.chz
class ResearchAgentSolver(BasePBSolver):
    """PaperBench solver with an explicit plan, checker, and final gate.

    This is a new solver class; ``BasicAgentSolver`` and ``IterativeAgent`` are
    not changed. It uses one compatible completer configuration for planning,
    primary execution, and each fresh Check Agent.
    """

    completer_config: Any = chz.field(default_factory=_default_compatible_config)
    computer_runtime: Any = chz.field(default_factory=_default_computer_runtime)
    max_steps: int | None = chz.field(default=None)
    planning_retries: int | None = chz.field(
        default=None,
        doc="Maximum invalid-plan retries; None keeps repairing until the JSON is valid",
    )
    plan_file: str | None = chz.field(
        default=None,
        doc=(
            "Path to a user-specified frozen plan JSON ({'subtasks':[...]}). "
            "When provided, the planner phase is skipped entirely and execution "
            "starts directly from this plan."
        ),
    )
    checker_max_steps: int = chz.field(default=300)
    mount_docker_socket: bool = chz.field(default=True)
    sanity_check_docker_enabled: bool = chz.field(default=False)
    # Code-Dev uses PaperBench's canonical code-only task instructions. The
    # copied ResearchAgent prompts below provide the stricter implementation
    # protocol without asking the agent to execute full experiments.
    iterative_agent: bool = chz.field(default=True)
    strict_reproduction_prompt: bool = chz.field(default=False)
    host_workspace_policy: bool = chz.field(default=False)
    workspace_policy: bool = chz.field(default=True)
    # Kept only for backward compatibility with pre-workspace run groups.
    skill_target: str = chz.field(default="")
    skill_library_root: str | None = chz.field(default=None)
    enable_skills: bool = chz.field(default=True)
    landlock_workspace_policy: bool = chz.field(default=False)

    @override
    def shortname(self) -> str:
        return "researchagent-skill" if self.enable_skills else "researchagent"

    def _default_agent_tools(self) -> list[Any]:
        from paperbench.solvers.basicagent.tools import BashTool, ReadFileChunk, SearchFile

        # Bash is for command execution. Source/configuration content is
        # written through the ResearchAgent's structured write_file tool so
        # code text is never mistaken for a filesystem path.
        return [BashTool(), SearchFile(), ReadFileChunk()]

    async def _run_agent(self, computer: Any, task: Any) -> Any:
        from paperbench.nano.structs import AgentOutput
        from paperbench.solvers.utils import check_for_existing_run

        existing = await check_for_existing_run(task)
        if existing is not None:
            return existing
        start_time = time.time()
        resolver = getattr(computer, "resolve_path", None)
        workspace = Path(str(resolver(_WORKSPACE_BASE) if callable(resolver) else _WORKSPACE_BASE))
        paper_root = workspace / "paper"
        store = PlanStore(Path(task.run_dir) / "research_agent")
        api_settings = CompatibleAPISettings.from_config(self.completer_config)
        from .runtime_checker import RuntimeCheckerFactory
        from .runtime_tools import ReadOnlyToolBundle, protocol_tools

        read_bundle = ReadOnlyToolBundle(
            public_root=str(paper_root),
            workspace_root=str(workspace),
            submission_root=str(workspace / "submission"),
            instructions_file=str(workspace / "instructions.txt"),
        )
        planner_tools = read_bundle.tools()
        planner = _ComputerToolLoopCompleter(
            config=api_settings.new_config(basicagent_tools=planner_tools),
            tools=planner_tools,
            computer=computer,
            tool_log=store.tool_log,
        )
        # The controller may resolve /home to a host bind-mount path, but the
        # agent-facing PaperBench filesystem contract remains the canonical
        # /home tree inside the task container.
        workspace_path = str(getattr(task, "agent_workspace", _WORKSPACE_BASE))
        planning_prompt = render_workspace_prompt(PLANNING_PROMPT, workspace_path)
        execution_prompt = render_workspace_prompt(
            execution_prompt_for_skills(enabled=self.enable_skills), workspace_path
        )
        session = ResearchAgentSession(
            store=store,
            completer=planner,
            planning_prompt=planning_prompt,
            execution_prompt=execution_prompt,
        )
        if store.plan_path.exists():
            session.window = MessageWindow(
                {"role": "system", "content": execution_prompt},
                render_plan_message(store.render()),
            )
            store.events.append("execution_resumed")
        elif self.plan_file:
            plan_content = json.loads(
                Path(self.plan_file).read_text(encoding="utf-8")
            )
            store.initialize(plan_content)
            session.window = MessageWindow(
                {"role": "system", "content": execution_prompt},
                render_plan_message(store.render()),
            )
            store.events.append("plan_provided_external", plan_file=self.plan_file)
        else:
            await session.create_plan(
                public_material_message=(
                    "The public task materials are mounted below. Read the complete files with "
                    "the read-only tools before planning:\n"
                    f"{workspace_path}/paper/paper.md\n{workspace_path}/paper/addendum.md\n"
                    f"{workspace_path}/paper/assets/\n"
                    f"{workspace_path}/instructions.txt\n"
                    f"Submit all implementation files under {workspace_path}/submission/. "
                    "The model/API is "
                    "configured by the runtime; do not inspect or expose credentials."
                ),
                max_retries=self.planning_retries,
            )

        job_manager = ManagedJobManager(store=store, computer=computer, workspace_root=workspace)
        # The checker-only briefing lives beside the protocol logs, outside the
        skill_coordinator: SkillCoordinator | None = None
        if self.enable_skills:
            if not self.skill_target:
                raise ValueError("skill_target is required when skills are enabled")
            skill_root = (
                Path(self.skill_library_root)
                if self.skill_library_root is not None
                else Path(__file__).resolve().parents[2] / "skill_library" / "paperbench"
            )
            catalog = SkillCatalog(root=skill_root, target=self.skill_target)
            skill_selector = SkillSelector(
                catalog=catalog,
                config=api_settings,
                read_only_tools=planner_tools,
                computer=computer,
                store=store,
                workspace_path=workspace_path,
            )
            skill_coordinator = SkillCoordinator(
                catalog=catalog,
                selector=skill_selector,
                store=store,
            )

        # agent-visible workspace, so the primary agent can never read it.
        checker_instructions_path = store.run_dir / "check_instruction.txt"
        if not checker_instructions_path.exists():
            checker_instructions_path.write_text(
                render_workspace_prompt(CHECK_INSTRUCTIONS, workspace_path), encoding="utf-8"
            )
        manager = ResearchAgentTools(
            store=store,
            runtime=_ComputerProcessRuntime(computer),
            checker_factory=RuntimeCheckerFactory(max_steps=self.checker_max_steps),
            completer_config=api_settings,
            public_paper_root=paper_root,
            workspace_root=workspace,
            submission_root=workspace / "submission",
            instructions_file=workspace / "instructions.txt",
            job_manager=job_manager,
            checker_instructions_file=checker_instructions_path,
            snapshotter=(
                task.snapshotter
                if getattr(task, "snapshotter", None) is not None
                else _PaperBenchSnapshotter(
                    computer=computer, task=task, start_time=start_time
                )
            ),
        )
        try:
            main_tools = self._default_agent_tools() + protocol_tools(manager)
            await self._execute_agent_loop(
                computer=computer,
                session=session,
                manager=manager,
                tools=main_tools,
                completer_config=api_settings.new_config(basicagent_tools=main_tools),
                skill_coordinator=skill_coordinator,
            )
        finally:
            await job_manager.close()
        end_time = time.time()
        return AgentOutput(
            run_id=task.run_id,
            time_start=start_time,
            time_end=end_time,
            error_msg=None,
            runtime_in_seconds=end_time - start_time,
            status_exists=bf.exists(bf.join(task.run_dir, "status.json")),
        )

    async def _execute_agent_loop(
        self,
        *,
        computer: Any,
        session: ResearchAgentSession,
        manager: ResearchAgentTools,
        tools: list[Any],
        completer_config: Any,
        skill_coordinator: SkillCoordinator | None,
    ) -> int:
        from .completer_resilience import (
            RepetitionTracker,
            build_repetition_warning,
            is_repetitive_tool_call_error,
        )
        from paperbench.solvers.basicagent.api import make_completer_request
        from paperbench.solvers.utils import log_messages_to_file

        if session.window is None:
            raise RuntimeError("execution context is missing")
        session.window.audit_log = session.store.tool_log
        completer = completer_config.build()
        context_window = getattr(completer, "context_window", None) or getattr(
            completer, "n_ctx", None
        )
        # Keep the canonical PaperBench rollout log at the run root so the
        # existing monitor/checkpoint/grade pipeline can find it. Protocol
        # events remain in the sibling research_agent directory.
        log_path = str(session.store.run_dir.parent / "agent.log")
        if skill_coordinator is not None:
            await skill_coordinator.ensure_current(session)
        log_messages_to_file(session.execution_messages(), log_path)
        by_name = {tool.name(): tool for tool in tools}
        repetition_tracker = RepetitionTracker(threshold=_REPETITION_WARNING_THRESHOLD)
        step = 0
        while self.max_steps is None or step < self.max_steps:
            step += 1
            session.sync_plan_message()
            if skill_coordinator is not None:
                await skill_coordinator.ensure_current(session)
            if context_window:
                before_tokens, after_tokens, removed_units = _trim_execution_context(
                    session=session,
                    completer=completer,
                    context_window=int(context_window),
                )
                if removed_units:
                    session.store.events.append(
                        "context_trim",
                        step=step,
                        estimated_tokens_before=before_tokens,
                        estimated_tokens_after=after_tokens,
                        removed_units=removed_units,
                        trigger_ratio=_CONTEXT_TRIM_TRIGGER_RATIO,
                        target_ratio=_CONTEXT_TRIM_TARGET_RATIO,
                    )
            try:
                response = await audited_request(
                    make_completer_request,
                    audit_log=session.store.tool_log,
                    role="main",
                    completer=completer, messages=session.window.messages
                )
            except (LengthFinishReasonError, IndexError):
                if context_window:
                    before_tokens, after_tokens, removed_units = _trim_execution_context(
                        session=session,
                        completer=completer,
                        context_window=int(context_window),
                        force=True,
                    )
                else:
                    removed_units = int(session.window.trim_oldest_unit())
                    before_tokens = after_tokens = None
                if not removed_units:
                    raise
                session.store.events.append(
                    "context_overflow_trim",
                    step=step,
                    estimated_tokens_before=before_tokens,
                    estimated_tokens_after=after_tokens,
                    removed_units=removed_units,
                )
                continue
            except Exception as exc:
                # Layer B: the provider rejected the request because the history
                # contained repeated identical tool calls. Drop the duplicate
                # units, explain the change, and retry instead of crashing.
                if not is_repetitive_tool_call_error(exc):
                    raise
                removed_units = session.window.drop_duplicate_tool_call_units()
                session.store.events.append(
                    "repetitive_tool_call_recovered",
                    step=step,
                    removed_units=removed_units,
                )
                if removed_units:
                    session.window.append(
                        {"role": "user", "content": _REPETITION_RECOVERY_MESSAGE}
                    )
                    repetition_tracker.reset()
                    log_messages_to_file(session.execution_messages(), log_path)
                    continue
                # Nothing removable (e.g. only the protected slots remain): the
                # retry would fail identically, so surface the provider error.
                raise
            session.window.extend(response.response_messages)
            log_messages_to_file(session.execution_messages(), log_path)
            if not response.tool_calls:
                current_task = session.store.first_uncompleted_task()
                if current_task is None:
                    continuation = (
                        "Please continue working. Every subtask is completed or skipped; "
                        "call final_submit when all final submission requirements are ready."
                    )
                else:
                    continuation = (
                        "Please continue working. Your current highest-priority unfinished "
                        f"subtask is {current_task.id}: {current_task.name}."
                    )
                session.window.append(
                    {
                        "role": "user",
                        "content": continuation,
                    }
                )
                log_messages_to_file(session.execution_messages(), log_path)
                continue
            finished = False
            for call in response.tool_calls:
                session.store.tool_log.append(
                    "agent_tool_call",
                    tool=call.name,
                    arguments=redact_for_log(dict(call.arguments)),
                )
                tool = by_name.get(call.name)
                if tool is None:
                    output = f"ERROR: unknown tool {call.name!r}"
                    session.store.tool_log.append(
                        "agent_tool_error", tool=call.name, error=output
                    )
                else:
                    try:
                        output = await tool.execute(computer, **call.arguments)
                    except Exception as exc:
                        output = f"ERROR: {type(exc).__name__}: {exc}"
                        session.store.tool_log.append(
                            "agent_tool_error", tool=call.name, error=output
                        )
                    else:
                        session.store.tool_log.append(
                            "agent_tool_result",
                            tool=call.name,
                            result=_log_value(redact_for_log(output)),
                        )
                session.window.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": str(output),
                    }
                )
                if repetition_tracker.observe(call):
                    session.window.append(
                        {
                            "role": "user",
                            "content": build_repetition_warning(
                                call, count=repetition_tracker.threshold
                            ),
                        }
                    )
                    session.store.events.append(
                        "repetitive_tool_call_warning",
                        step=step,
                        tool=call.name,
                        count=repetition_tracker.threshold,
                    )
                if call.name == "final_submit":
                    try:
                        finished = bool(json.loads(str(output)).get("accepted"))
                    except (TypeError, ValueError):
                        finished = False
                if finished:
                    break
            log_messages_to_file(session.execution_messages(), log_path)
            if finished and manager.state.phase == "evaluation":
                return step
        raise RuntimeError("ResearchAgent reached max_steps before final_submit was accepted")


class _ComputerToolLoopCompleter:
    """Completer adapter that lets a planning model use only read-only tools."""

    def __init__(self, *, config: Any, tools: list[Any], computer: Any, tool_log: Any | None = None):
        self.config = config
        self.tools = tools
        self.computer = computer
        self.tool_log = tool_log

    async def complete(self, messages: Sequence[dict[str, Any]], *, tools: Sequence[Any]) -> str:
        from paperbench.solvers.basicagent.api import make_completer_request

        del tools
        if len(messages) < 2:
            raise ValueError("planning context must contain system and public-material messages")
        window = MessageWindow(messages[0], messages[1])
        window.audit_log = self.tool_log
        window.extend(messages[2:])
        by_name = {tool.name(): tool for tool in self.tools}
        completer = self.config.build()
        for _ in range(64):
            try:
                response = await audited_request(
                    make_completer_request,
                    audit_log=self.tool_log,
                    role="planner",
                    completer=completer, messages=window.messages
                )
            except (LengthFinishReasonError, IndexError):
                if not window.trim_oldest_unit():
                    raise
                if self.tool_log is not None:
                    self.tool_log.append("planning_context_trim")
                continue
            window.extend(response.response_messages)
            if not response.tool_calls:
                content = window.messages[-1].get("content", "")
                return content if isinstance(content, str) else str(content)
            for call in response.tool_calls:
                if self.tool_log is not None:
                    self.tool_log.append(
                        "planning_tool_call",
                        tool=call.name,
                        arguments=dict(call.arguments),
                    )
                tool = by_name.get(call.name)
                if tool is None:
                    output = f"ERROR: unsupported planning tool {call.name!r}"
                    if self.tool_log is not None:
                        self.tool_log.append(
                            "planning_tool_error", tool=call.name, error=output
                        )
                else:
                    try:
                        output = await tool.execute(self.computer, **call.arguments)
                    except Exception as exc:
                        output = f"ERROR: {type(exc).__name__}: {exc}"
                        if self.tool_log is not None:
                            self.tool_log.append(
                                "planning_tool_error", tool=call.name, error=output
                            )
                    else:
                        if self.tool_log is not None:
                            self.tool_log.append(
                                "planning_tool_result",
                                tool=call.name,
                                result=_log_value(output),
                            )
                window.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": str(output),
                    }
                )
        raise RuntimeError("Planning Agent exceeded its read-only tool-call limit")


class _ComputerProcessRuntime:
    def __init__(self, computer: Any):
        self.computer = computer

    async def active_processes(self) -> list[AgentProcess]:
        scanner = getattr(self.computer, "active_processes", None)
        if not callable(scanner):
            return []
        values = await scanner()
        return [AgentProcess(int(item.pid), str(item.command)) for item in values]


class _PaperBenchSnapshotter(Snapshotter):
    """Freeze through PaperBench's normal upload/checkpoint path."""

    def __init__(self, *, computer: Any, task: Any, start_time: float):
        self.computer = computer
        self.task = task
        self.start_time = start_time

    async def freeze_submission(self) -> str:
        from paperbench.constants import AGENT_DIR_CONFIG
        from paperbench.solvers.upload import upload_heavy_logs, upload_status

        await upload_heavy_logs(
            computer=self.computer,
            agent_start_time=int(self.start_time),
            agent_dir_config=AGENT_DIR_CONFIG,
            run_dir=self.task.run_dir,
            run_group_id=self.task.run_group_id,
            runs_dir=self.task.runs_dir,
            run_id=self.task.run_id,
        )
        await upload_status(
            start_time=int(self.start_time),
            run_dir=self.task.run_dir,
            status="done",
            end_time=int(time.time()),
        )
        archives = sorted(
            bf.glob(bf.join(self.task.run_dir, "submissions", "*", "submission.tar.gz"))
        )
        if not archives:
            raise FileNotFoundError("PaperBench upload produced no submission.tar.gz")
        return str(archives[-1])
