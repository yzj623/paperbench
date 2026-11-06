"""Dependency-free local transport for the original main-agent tool contracts."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from .state_machine import ResearchToolError


class WorkspaceWriteFileTool:
    """The RA write_file path/content contract, without a PaperBench Tool base."""

    def __init__(self, submission_root: Path):
        self.submission_root = submission_root

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
                raw = raw[len(virtual_root) + 1:]
                break
        candidate = Path(raw)
        if ".." in candidate.parts:
            raise ResearchToolError("PATH_ESCAPE", "path may not contain '..'", path=path)
        root = self.submission_root.resolve(strict=False)
        lexical = Path(os.path.abspath(raw)) if candidate.is_absolute() else root / candidate
        if not self._within(lexical, root):
            raise ResearchToolError("PATH_ESCAPE", "write_file path must stay under the submission directory", path=path)
        resolved = lexical.resolve(strict=False)
        if not self._within(resolved, root) or lexical.is_symlink():
            raise ResearchToolError("PATH_ESCAPE", "write_file path escapes through a symlink", path=path)
        if lexical.exists() and not lexical.is_file():
            raise ResearchToolError("INVALID_ARGUMENT", "write_file target must be a regular file", path=path)
        return lexical

    async def execute(self, computer: object, *, path: str, content: str) -> str:
        del computer
        try:
            if not isinstance(content, str):
                raise ResearchToolError("INVALID_ARGUMENT", "content must be a string")
            target = self._resolve_target(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not self._within(target.parent.resolve(strict=True), self.submission_root.resolve(strict=False)):
                raise ResearchToolError("PATH_ESCAPE", "write_file parent escapes the submission directory", path=path)
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
            return json.dumps({"ok": False, "error": {"code": exc.code, "message": str(exc),
                                                      "details": exc.details}}, ensure_ascii=False)
        return json.dumps({"ok": True, "path": target.as_posix(), "bytes_written": len(encoded)},
                          ensure_ascii=False)


class BashTool:
    async def execute(self, computer: object, *, cmd: str) -> str:
        result = await computer.send_shell_command(cmd)
        return result.output.decode("utf-8", "replace").strip()


class ReadFileChunk:
    async def execute(self, computer: object, *, file: str, start_line: int = 1,
                      max_lines: int = 50) -> str:
        if start_line < 1:
            return "ERROR: start_line must be >= 1"
        if max_lines < 1:
            return "ERROR: max_lines must be >= 1"
        if max_lines > 50:
            return "ERROR: max_lines cannot exceed 50"
        try:
            result = await computer.send_shell_command(f"cat {file}")
            lines = result.output.decode("utf-8").strip().splitlines()
            if start_line > len(lines):
                return (f"ERROR: start_line ({start_line}) is beyond the total number of lines "
                        f"({len(lines)}) in the file.")
            end_line = min(start_line + max_lines - 1, len(lines))
            chunk = lines[start_line - 1:end_line]
            return (f"File has {len(lines)} total lines. Showing lines {start_line} to {end_line}.\n\n" +
                    "\n".join(f"{i + start_line}: {line}" for i, line in enumerate(chunk)))
        except FileNotFoundError:
            return f"ERROR: File '{file}' not found"
        except Exception as exc:
            return f"ERROR: Error reading file: {exc}"


class SearchFile:
    async def execute(self, computer: object, *, file: str, query: str,
                      context_lines: int = 2, max_matches: int = 5, page: int = 1) -> str:
        if not query:
            return "ERROR: Query cannot be empty."
        if context_lines < 0:
            return "ERROR: context_lines must be >= 0"
        if max_matches < 1:
            return "ERROR: max_matches must be >= 1"
        if page < 1:
            return "ERROR: page must be >= 1"
        try:
            result = await computer.send_shell_command(f"cat {file}")
            lines = result.output.decode("utf-8").strip().splitlines()
            matches = []
            for index, line in enumerate(lines):
                if query.lower() in line.lower():
                    matches.append("\n".join(
                        f"{'>>> ' if i == index else '    '}{i + 1}: {lines[i]}"
                        for i in range(max(0, index - context_lines), min(len(lines), index + context_lines + 1))
                    ))
            if not matches:
                return f"No matches found for '{query.lower()}' in {file}"
            total = len(matches)
            pages = (total + max_matches - 1) // max_matches
            if page > pages:
                return f"Invalid page number. There are only {pages} pages of results."
            start = (page - 1) * max_matches
            end = min(start + max_matches, total)
            summary = [f"Found {total} matches for '{query.lower()}' in {file}",
                       f"Showing matches {start + 1}-{end} (Page {page} of {pages})", ""]
            return "\n\n".join(summary + [f"[Match {i + 1} of {total}]\n{matches[i]}" for i in range(start, end)])
        except FileNotFoundError:
            return f"ERROR: File '{file}' not found"
        except Exception as exc:
            return f"ERROR: Error searching file: {exc}"
