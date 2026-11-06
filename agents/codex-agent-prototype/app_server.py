"""Async JSONL client for the standalone Codex app server (no SDK or login)."""
from __future__ import annotations

import asyncio
import json
import os
import signal
from pathlib import Path
from typing import Any
from protocol.async_jobs import redact_for_log


class AppServerError(RuntimeError):
    pass


def server_command() -> list[str]:
    raw = os.environ.get("CODEX_APP_SERVER_COMMAND")
    command = json.loads(raw) if raw else ["/opt/codex/bin/codex-app-server"]
    if not isinstance(command, list) or not command or any(
        not isinstance(item, str) or not item or "\0" in item for item in command
    ):
        raise ValueError("CODEX_APP_SERVER_COMMAND must be a non-empty JSON argv array")
    return command


class AppServer:
    """One reader owns stdout; responses bypass the sequential event consumer.

    In particular a tool handler may start a checker or issue another RPC while
    the reader continues draining main's stdout. Never read stdout elsewhere.
    Raw streams and provider error bodies are deliberately not persisted.
    """

    def __init__(self, *, cwd: Path, codex_home: Path, command=None, env=None,
                 request_timeout: float = 30, trace_path: Path | None = None):
        self.cwd, self.codex_home = cwd, codex_home
        self.command = list(command if command is not None else server_command())
        for flag in ("--ppblab-protocol", "--strict-config"):
            if flag not in self.command:
                self.command.append(flag)
        self.env = dict(os.environ if env is None else env)
        self.env["CODEX_HOME"] = str(codex_home)
        self.request_timeout = request_timeout
        self.events: asyncio.Queue = asyncio.Queue()
        self.pending: dict[int, asyncio.Future] = {}
        self.serial = 0
        self.process = None
        self.reader = None
        self.stderr_reader = None
        self.write_lock = asyncio.Lock()
        self.stderr_tail = ""
        self.trace_path = trace_path

    def record(self, message):
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.trace_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(redact_for_log(message), ensure_ascii=False) + "\n")

    async def __aenter__(self):
        self.codex_home.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.command, cwd=self.cwd, env=self.env,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, start_new_session=True,
                limit=16 * 1024 * 1024)
            self.reader = asyncio.create_task(self._read())
            self.stderr_reader = asyncio.create_task(self._drain_stderr())
            await self.request("initialize", {
                "clientInfo": {"name": "ppblab-controller", "version": "1"},
                "capabilities": {"experimentalApi": True}})
            await self.send({"method": "initialized"})
            return self
        except BaseException:
            await self.close()
            raise

    async def __aexit__(self, *args):
        await self.close()

    async def _drain_stderr(self):
        # Keep a bounded, sanitized diagnostic tail; never save raw stderr.
        while chunk := await self.process.stderr.readline():
            self.stderr_tail = (self.stderr_tail + redact_for_log(
                chunk.decode("utf-8", "replace")))[-4096:]

    async def _read(self):
        failure = AppServerError("app-server stdout closed")
        try:
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("expected JSON object")
                self.record(message)
                if "id" in message and "method" not in message:
                    future = self.pending.get(message["id"])
                    if future is not None and not future.done():
                        if "error" in message:
                            diagnostic = redact_for_log(message["error"])
                            future.set_exception(AppServerError(
                                "app-server RPC rejected: " + str(diagnostic)[:4096]))
                        elif "result" in message:
                            future.set_result(message["result"])
                        else:
                            future.set_exception(AppServerError("malformed RPC response"))
                else:
                    await self.events.put(message)
        except asyncio.CancelledError:
            return
        except Exception:
            failure = AppServerError("invalid app-server JSONL stream")
        finally:
            if self.stderr_tail:
                failure = AppServerError(str(failure) + ": " + self.stderr_tail)
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(failure)
            await self.events.put(failure)

    async def send(self, message: dict):
        async with self.write_lock:
            if self.process is None or self.process.returncode is not None:
                raise AppServerError("app-server is not running")
            try:
                self.process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
                await self.process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise AppServerError("app-server input closed") from exc

    async def request(self, method: str, params: dict | None = None):
        self.serial += 1
        identifier = self.serial
        future = asyncio.get_running_loop().create_future()
        self.pending[identifier] = future
        try:
            await self.send({"id": identifier, "method": method, "params": params or {}})
            return await asyncio.wait_for(future, self.request_timeout)
        finally:
            self.pending.pop(identifier, None)

    async def set_prefix(self, thread_id: str, messages: list[dict]):
        """Atomically replace the thread's request-only messages; [] clears them."""
        return await self.request("thread/prefix/set", {"threadId": thread_id, "messages": messages})

    async def respond(self, identifier: Any, text: str, *, success: bool = True):
        response = {"id": identifier, "result": {
            "contentItems": [{"type": "inputText", "text": text}], "success": success}}
        await self.send(response)
        self.record({"direction": "client-tool-response", **response})

    async def next_event(self):
        value = await self.events.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def resolve_model(self, explicit: str | None = None) -> str:
        if explicit:
            return explicit
        cursor = None
        seen = set()
        fallback = None
        while True:
            page = await self.request("model/list", {"cursor": cursor, "limit": 100})
            for entry in page.get("data", []):
                if entry.get("hidden"):
                    continue
                model = entry.get("model") or entry.get("id")
                if model:
                    fallback = fallback or model
                    if entry.get("isDefault"):
                        return model
            cursor = page.get("nextCursor")
            if not cursor:
                break
            if cursor in seen:
                raise AppServerError("model/list repeated its cursor")
            seen.add(cursor)
        if not fallback:
            raise AppServerError("model/list returned no available model")
        return fallback

    async def close(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self.process.wait()
        for task in (self.reader, self.stderr_reader):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in (self.reader, self.stderr_reader)
                               if task is not None), return_exceptions=True)
