"""Read-only Check Agent tools with a realpath allowlist."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class PathAccessError(PermissionError):
    pass


class ReadOnlyWorkspace(Protocol):
    def list_directory(self, path: str) -> list[str]: ...

    def read_file(self, path: str, *, start_line: int = 1, max_lines: int = 200) -> str: ...

    def search_file(self, path: str, *, query: str, context_lines: int = 2) -> str: ...


class SafeWorkspace:
    """Allow access only below explicit public-paper/submission roots.

    The evaluator's rubric, judge, reference, and grade paths are denied even
    if a caller accidentally includes one of those names under an allowed
    root.  Symlinks are resolved before access and must remain inside a root.
    """

    _PRIVATE_COMPONENTS = {
        "rubric",
        "rubric.json",
        "judge",
        "reference",
        "references",
        "grade",
        "grade.json",
        "grades",
        "gold",
        ".env",
        "agent.env",
        "credential",
        "credentials",
        "secret",
        "secrets",
    }

    def __init__(
        self,
        *,
        public_root: str | Path,
        workspace_root: str | Path,
        submission_root: str | Path | None = None,
        instructions_file: str | Path | None = None,
        checker_instructions_file: str | Path | None = None,
        checker_submission_only: bool = False,
    ):
        # These are physical host paths.  The agents see the stable PaperBench
        # paths below, which HostUV translates into this same runtime root.
        self.public_root = Path(public_root).resolve(strict=True)
        self.workspace_root = Path(workspace_root).resolve(strict=True)
        self.submission_root = Path(
            submission_root if submission_root is not None else self.workspace_root / "submission"
        ).resolve()
        if checker_instructions_file is not None:
            # Checker context: the agent-facing instructions file is not part
            # of the evidence set; a dedicated briefing file is exposed at the
            # virtual path /check_instruction.txt instead.
            self.instructions_file: Path | None = None
            self.checker_instructions_file = Path(checker_instructions_file).resolve(strict=True)
        else:
            self.checker_instructions_file = None
            self.instructions_file = (
                Path(instructions_file) if instructions_file is not None else self.workspace_root / "instructions.txt"
            ).resolve()
        if checker_submission_only and self.checker_instructions_file is None:
            raise ValueError("checker_submission_only requires checker_instructions_file")
        self.checker_submission_only = checker_submission_only
        # The checker must not be able to enumerate all of /home.  Only the
        # public paper tree, submission tree, and one instructions/briefing
        # file are roots of the read-only allowlist.
        roots: list[Path] = [self.submission_root]
        if not checker_submission_only:
            roots.insert(0, self.public_root)
        if self.checker_instructions_file is not None and not checker_submission_only:
            roots.append(self.checker_instructions_file)
        elif self.instructions_file is not None:
            roots.append(self.instructions_file)
        self._roots = tuple(roots)

    def _is_private(self, path: Path) -> bool:
        return any(component.lower() in self._PRIVATE_COMPONENTS for component in path.parts)

    def resolve(self, requested: str | Path, *, directory: bool = False) -> Path:
        raw_requested = str(requested).replace("\\", "/")
        candidate = Path(requested)
        if ".." in candidate.parts or ".." in Path(raw_requested).parts:
            raise PathAccessError("parent traversal is not allowed")
        # Relative paths are interpreted relative to workspace, never cwd.  An
        # absolute virtual path is translated explicitly so a checker can use
        # the same /home/paper and /home/submission paths as the agent.
        is_virtual_home = raw_requested == "/home" or raw_requested.startswith("/home/")
        is_virtual_submission = raw_requested == "/submission" or raw_requested.startswith(
            "/submission/"
        )
        if not candidate.is_absolute() and not (is_virtual_home or is_virtual_submission):
            # Relative evidence paths are relative to the submitted project,
            # never to the broad /home workspace.
            candidate = self.submission_root / candidate
        else:
            virtual = raw_requested
            if is_virtual_home:
                suffix = virtual[len("/home") :].lstrip("/")
                if suffix == "paper" or suffix.startswith("paper/"):
                    suffix = suffix[len("paper") :].lstrip("/")
                    candidate = self.public_root / suffix
                else:
                    if suffix == "instructions.txt":
                        if self.instructions_file is None:
                            raise PathAccessError(
                                "the agent-facing task instructions are not part of the "
                                "Check Agent evidence set"
                            )
                        candidate = self.instructions_file
                    else:
                        candidate = self.workspace_root / suffix
            elif virtual == "/check_instruction.txt":
                if self.checker_instructions_file is None:
                    raise PathAccessError("the checker briefing is only available to the Check Agent")
                candidate = self.checker_instructions_file
            elif is_virtual_submission:
                suffix = virtual[len("/submission") :].lstrip("/")
                candidate = self.submission_root / suffix
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise PathAccessError(f"path does not exist: {requested}") from exc
        if self._is_private(resolved):
            raise PathAccessError("evaluator-private path is not accessible")
        if directory and not resolved.is_dir():
            raise PathAccessError("requested path is not a directory")
        if not directory and not resolved.is_file():
            raise PathAccessError("requested path is not a regular file")
        if not any(resolved == root or root in resolved.parents for root in self._roots):
            raise PathAccessError("path escapes the read-only allowlist")
        # An explicit symlink is allowed only when its resolved target stayed in
        # the allowlist; this also blocks ../ and symlink traversal escapes.
        if candidate.is_symlink() and not any(
            resolved == root or root in resolved.parents for root in self._roots
        ):
            raise PathAccessError("symlink escapes the read-only allowlist")
        return resolved

    def list_directory(self, path: str) -> list[str]:
        resolved = self.resolve(path, directory=True)
        visible: list[str] = []
        for child in sorted(resolved.iterdir()):
            if child.name.lower() in self._PRIVATE_COMPONENTS:
                continue
            # Do not expose a child that itself resolves outside the root.
            try:
                self.resolve(child, directory=child.is_dir())
            except PathAccessError:
                continue
            visible.append(child.name)
        return visible

    def read_file(self, path: str, *, start_line: int = 1, max_lines: int = 200) -> str:
        if start_line < 1 or max_lines < 1:
            raise ValueError("start_line and max_lines must be positive")
        resolved = self.resolve(path)
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        if start_line > len(lines):
            return f"File has {len(lines)} total lines; start_line is beyond EOF."
        end = min(start_line + max_lines - 1, len(lines))
        return (
            f"File has {len(lines)} total lines. Showing lines {start_line} to {end}.\n\n"
            + "\n".join(f"{number}: {lines[number - 1]}" for number in range(start_line, end + 1))
        )

    def search_file(self, path: str, *, query: str, context_lines: int = 2) -> str:
        if not query:
            raise ValueError("query cannot be empty")
        if context_lines < 0:
            raise ValueError("context_lines must be non-negative")
        resolved = self.resolve(path)
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        matches: list[str] = []
        needle = query.casefold()
        for index, line in enumerate(lines):
            if needle in line.casefold():
                start = max(0, index - context_lines)
                end = min(len(lines), index + context_lines + 1)
                matches.append(
                    "\n".join(
                        f"{'>>> ' if offset == index else '    '}{offset + 1}: {lines[offset]}"
                        for offset in range(start, end)
                    )
                )
        return f"Found {len(matches)} matches for {query!r}.\n\n" + "\n\n".join(matches)


    @staticmethod
    def _text_lines(path: Path) -> list[str]:
        data = path.read_bytes()
        try:
            if b"\x00" in data:
                raise ValueError("binary content")
            return data.decode("utf-8").splitlines()
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(
                "This tool supports UTF-8 text only; binary/image content cannot be read as text."
            ) from exc


class ReadOnlyToolDispatcher:
    """Dispatch exactly the three safe tools exposed to a Check Agent."""

    def __init__(self, workspace: SafeWorkspace):
        self.workspace = workspace
        # Keep the schemas next to the dispatcher so a Checker can use this
        # object for both model tool definitions and safe dispatch.
        self.schemas = {
            item["function"]["name"]: item for item in readonly_tool_schemas()
        }

    @staticmethod
    def _bounded_integer(
        value: Any, *, name: str, minimum: int, maximum: int | None = None
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise PathAccessError(f"{name} must be an integer")
        if value < minimum or (maximum is not None and value > maximum):
            limit = f" between {minimum} and {maximum}" if maximum is not None else f" >= {minimum}"
            raise PathAccessError(f"{name} must be{limit}")
        return value

    def call(self, name: str, arguments: dict[str, Any]) -> str | list[str]:
        if name == "list_directory":
            return self.workspace.list_directory(arguments["path"])
        if name == "read_file":
            start_line = self._bounded_integer(
                arguments.get("start_line", 1), name="start_line", minimum=1
            )
            max_lines = self._bounded_integer(
                arguments.get("max_lines", 200), name="max_lines", minimum=1, maximum=200
            )
            return self.workspace.read_file(
                arguments["path"],
                start_line=start_line,
                max_lines=max_lines,
            )
        if name == "search_file":
            context_lines = self._bounded_integer(
                arguments.get("context_lines", 2),
                name="context_lines",
                minimum=0,
                maximum=20,
            )
            return self.workspace.search_file(
                arguments["path"],
                query=arguments["query"],
                context_lines=context_lines,
            )
        raise PathAccessError(f"unsupported Check Agent tool: {name}")


def readonly_tool_schemas() -> list[dict[str, Any]]:
    """The only three tool definitions exposed to a Check Agent."""

    return [
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List entries under an allowed public-paper or workspace directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read numbered lines from an allowed file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "max_lines": {"type": "integer", "minimum": 1, "maximum": 200},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_file",
                "description": "Search an allowed file and return matching lines with context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "query": {"type": "string"},
                        "context_lines": {"type": "integer", "minimum": 0, "maximum": 20},
                    },
                    "required": ["path", "query"],
                    "additionalProperties": False,
                },
            },
        },
    ]
