"""PaperBench tool adapters used by the planner, checker, and main agent."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable

from openai.types.responses import FunctionToolParam

from paperbench.solvers.basicagent.tools.base import Tool

from .readonly import ReadOnlyToolDispatcher, SafeWorkspace
from .state_machine import FinalSubmitResult, ResearchAgentTools, ResearchToolError


class ReadOnlyAgentTool(Tool):
    """A BasicAgent-compatible read-only tool backed by a SafeWorkspace."""

    tool_name: str
    description: str
    dispatcher: Any

    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return self.tool_name

    def get_oai_tool_call(self) -> FunctionToolParam:
        # ``ReadOnlyToolDispatcher.schemas`` is keyed by function name.  Keep
        # this adapter independent of the order/shape of that mapping so the
        # checker can pass valid OpenAI-compatible function definitions.
        function = dict(self.dispatcher.schemas[self.tool_name]["function"])
        function["description"] = self.description
        return FunctionToolParam(
            type="function",
            name=self.tool_name,
            description=self.description,
            parameters=function["parameters"],
            strict=False,
        )

    async def execute(self, computer: Any, **kwargs: Any) -> str:
        del computer
        return str(self.dispatcher.call(self.tool_name, kwargs))


class ReadOnlyToolBundle:
    """Builds read-only tools for paper inputs and the submission root."""

    def __init__(
        self,
        *,
        public_root: str,
        workspace_root: str,
        submission_root: str | None = None,
        instructions_file: str | None = None,
    ):
        self.workspace = SafeWorkspace(
            public_root=public_root,
            workspace_root=workspace_root,
            submission_root=submission_root,
            instructions_file=instructions_file,
        )
        self.dispatcher = ReadOnlyToolDispatcher(self.workspace)
        self.dispatcher.schemas = {
            item["function"]["name"]: item for item in self._schemas()
        }

    @staticmethod
    def _schemas() -> list[dict[str, Any]]:
        from .readonly import readonly_tool_schemas

        return readonly_tool_schemas()

    def tools(self) -> list[ReadOnlyAgentTool]:
        descriptions = {
            "list_directory": "List entries under an allowed public-paper or workspace directory.",
            "read_file": "Read numbered lines from an allowed file.",
            "search_file": "Search an allowed file and return matching lines with context.",
        }
        return [
            ReadOnlyAgentTool(
                tool_name=name,
                description=descriptions[name],
                dispatcher=self.dispatcher,
            )
            for name in descriptions
        ]


class ProtocolTool(Tool):
    """Adapter exposing one state-machine operation as a model function."""

    tool_name: str
    description: str
    parameters: dict[str, Any]
    manager: Any
    handler: Callable[..., Awaitable[Any]]

    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return self.tool_name

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.tool_name,
            description=self.description,
            parameters=self.parameters,
            strict=False,
        )

    async def execute(self, computer: Any, **kwargs: Any) -> str:
        del computer
        try:
            value = await self.handler(**kwargs)
        except ResearchToolError as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": exc.code,
                        "message": str(exc),
                        "details": exc.details,
                    },
                },
                ensure_ascii=False,
            )
        if isinstance(value, FinalSubmitResult):
            return json.dumps(
                {
                    "accepted": value.accepted,
                    "snapshot_path": value.snapshot_path,
                    "errors": list(value.errors),
                },
                ensure_ascii=False,
            )
        return str(value)


class WorkspaceWriteFileTool(Tool):
    """Atomically write one UTF-8 file below the task submission directory.

    The content is deliberately opaque. Only the explicit destination path
    is validated, so source-code operators and strings can never be confused
    with filesystem paths.
    """

    submission_root: Path

    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return "write_file"

    @staticmethod
    def _within(candidate: Path, root: Path) -> bool:
        return candidate == root or root in candidate.parents

    def _resolve_target(self, path: str) -> Path:
        if not isinstance(path, str) or not path or "\0" in path:
            raise ResearchToolError("INVALID_ARGUMENT", "path must be a non-empty string")
        raw = path.replace("\\", "/")
        for virtual_root in ("/home/submission", "/submission"):
            if raw == virtual_root:
                raw = "."
                break
            if raw.startswith(virtual_root + "/"):
                raw = raw[len(virtual_root) + 1 :]
                break
        candidate = Path(raw)
        if ".." in candidate.parts:
            raise ResearchToolError("PATH_ESCAPE", "path may not contain '..'", path=path)

        root = self.submission_root.resolve(strict=False)
        lexical = Path(os.path.abspath(raw)) if candidate.is_absolute() else root / candidate
        if not self._within(lexical, root):
            raise ResearchToolError(
                "PATH_ESCAPE", "write_file path must stay under the submission directory", path=path
            )
        try:
            resolved = lexical.resolve(strict=False)
        except OSError as exc:
            raise ResearchToolError("PATH_ESCAPE", f"cannot resolve write path: {path}") from exc
        if not self._within(resolved, root):
            raise ResearchToolError(
                "PATH_ESCAPE", "write_file path escapes through a symlink", path=path
            )
        if lexical.is_symlink():
            raise ResearchToolError("PATH_ESCAPE", "write_file target cannot be a symlink", path=path)
        if lexical.exists() and not lexical.is_file():
            raise ResearchToolError(
                "INVALID_ARGUMENT", "write_file target must be a regular file", path=path
            )
        return lexical

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description=(
                "Create or completely replace one UTF-8 source, configuration, script, or text "
                "file under the task submission directory. The path is validated, while content "
                "is treated as opaque text and is never interpreted as a filesystem path. Parent "
                "directories are created automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "A real absolute path below the task submission directory, or a path "
                            "relative to that directory."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete UTF-8 file content to write.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, computer: Any, *, path: str, content: str) -> str:
        del computer
        try:
            if not isinstance(content, str):
                raise ResearchToolError("INVALID_ARGUMENT", "content must be a string")
            target = self._resolve_target(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            parent = target.parent.resolve(strict=True)
            root = self.submission_root.resolve(strict=False)
            if not self._within(parent, root):
                raise ResearchToolError(
                    "PATH_ESCAPE", "write_file parent escapes the submission directory", path=path
                )

            existing_mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o644
            encoded = content.encode("utf-8")
            fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, existing_mode)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        except ResearchToolError as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": exc.code,
                        "message": str(exc),
                        "details": exc.details,
                    },
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {"ok": True, "path": target.as_posix(), "bytes_written": len(encoded)},
            ensure_ascii=False,
        )


def protocol_tools(manager: ResearchAgentTools) -> list[Tool]:
    job_runtime = manager.job_manager

    async def missing_job_runtime(**kwargs: Any) -> str:
        del kwargs
        raise ResearchToolError(
            "ASYNC_JOBS_UNAVAILABLE", "managed background tools are not attached to this runtime"
        )

    result = [
        WorkspaceWriteFileTool(
            submission_root=manager.readonly_workspace.submission_root,
        ),
        ProtocolTool(
            tool_name="run_background",
            description=(
                "Start a long-running task with an exec-style argv vector and return immediately. "
                "Output is redirected to the requested task-local log file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "cwd": {"type": "string"},
                    "env": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "log_path": {"type": "string"},
                },
                "required": ["argv", "cwd", "log_path"],
                "additionalProperties": False,
            },
            manager=manager,
            handler=job_runtime.run_background if job_runtime is not None else missing_job_runtime,
        ),
        ProtocolTool(
            tool_name="job_status",
            description="Inspect one managed background job, or list all managed jobs.",
            parameters={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "additionalProperties": False,
            },
            manager=manager,
            handler=job_runtime.job_status if job_runtime is not None else missing_job_runtime,
        ),
        ProtocolTool(
            tool_name="job_log",
            description="Read a bounded tail of one managed background job's log.",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "tail_lines": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
            manager=manager,
            handler=job_runtime.job_log if job_runtime is not None else missing_job_runtime,
        ),
        ProtocolTool(
            tool_name="job_kill",
            description="Stop one managed background job by job_id after identity validation.",
            parameters={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
                "additionalProperties": False,
            },
            manager=manager,
            handler=job_runtime.job_kill if job_runtime is not None else missing_job_runtime,
        ),
        ProtocolTool(
            tool_name="skip_subtask",
            description=(
                "Temporarily defer a genuinely impossible subtask only after it has received "
                "at least three valid Check Agent rejections. Earlier calls leave it uncompleted."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "subtask_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["subtask_id", "reason"],
                "additionalProperties": False,
            },
            manager=manager,
            handler=manager.skip_subtask,
        ),
        ProtocolTool(
            tool_name="submit_subtask",
            description="Submit one subtask to a fresh stateless Check Agent.",
            parameters={
                "type": "object",
                "properties": {
                    "subtask_id": {"type": "string"},
                    "reproduction_description": {"type": "string"},
                },
                "required": ["subtask_id", "reproduction_description"],
                "additionalProperties": False,
            },
            manager=manager,
            handler=manager.submit_subtask,
        ),
        ProtocolTool(
            tool_name="final_submit",
            description=("Save a complete submission snapshot and enter full review. During a review "
                         "response round, every issue must first be answered by respond_to_review. "
                         "The main-agent conversation continues after successful submission."
                         if manager.review_store is not None and manager.state.phase != "execution" else (
                "Finish only when every subtask is completed or validly skipped and all required "
                "files and process gates are ready. Uncompleted tasks are always rejected. If "
                "skipped tasks remain, the first otherwise-valid call asks for confirmation and "
                "a later call confirms evaluation with skipped work."
            )),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            manager=manager,
            handler=manager.final_submit,
        ),
    ]
    if manager.review_store is not None and manager.state.phase != "execution":
        result = [tool for tool in result if tool.name() != "skip_subtask"]
        result.append(ProtocolTool(
            tool_name="respond_to_review",
            description=("Respond exactly once to one issue from the current full review after checking "
                         "the paper and current code. Decide plan_inaccurate, checker_inaccurate, or "
                         "valid_issue, and give a concrete reason and evidence. Every issue must be "
                         "answered before final_submit; the tool does not change code."),
            parameters={"type": "object", "properties": {
                "issue_id": {"type": "string"},
                "decision": {"type": "string", "enum": ["plan_inaccurate", "checker_inaccurate", "valid_issue"]},
                "reason": {"type": "string"}, "evidence": {"type": "string"}},
                "required": ["issue_id", "decision", "reason", "evidence"],
                "additionalProperties": False},
            manager=manager,
            handler=manager.respond_to_review,
        ))
    return result
