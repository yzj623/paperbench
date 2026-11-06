"""Passive audit records; never feeds observations back into agent decisions."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import time
import uuid
import difflib
from pathlib import Path
from .storage import JsonlLog, atomic_json_write, redact_sensitive

def serial(value):
    if dataclasses.is_dataclass(value):
        return serial(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return serial(value.model_dump())
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return serial(vars(value))
    return str(value)

def record(log, event, **data):
    """Logging failures must not change reproduction or approval behavior."""
    if log is None:
        return
    try:
        JsonlLog(log.path.parent / "audit" / "events.jsonl").append(event, **serial(data))
    except Exception as exc:
        try:
            log.append("audit_error", operation=event, error_type=type(exc).__name__)
        except Exception:
            pass

class _AuditedCompleter:
    def __init__(self, target, log, role, request_id):
        self.target, self.log, self.role, self.request_id = target, log, role, request_id
        self.attempt = 0

    def __getattr__(self, name):
        return getattr(self.target, name)

    async def async_completion(self, conversation, **params):
        self.attempt += 1
        fields = {"request_id": self.request_id, "role": self.role, "attempt": self.attempt}
        record(self.log, "completion_attempt", **fields, messages=conversation, params=params)
        try:
            result = await self.target.async_completion(conversation=conversation, **params)
        except BaseException as exc:
            record(self.log, "completion_attempt_error", **fields,
                   error_type=type(exc).__name__, status_code=getattr(exc, "status_code", None))
            raise
        record(self.log, "completion_attempt_response", **fields, completion=serial(result))
        return result


async def audited_request(request, *, audit_log, role, **kwargs):
    request_id = uuid.uuid4().hex
    start = time.monotonic()
    messages = kwargs.get("messages", [])
    completer = kwargs.get("completer")
    snapshot_id = None
    if audit_log is not None:
        latest = audit_log.path.parent / "audit/latest.json"
        if latest.exists():
            try:
                snapshot_id = json.loads(latest.read_text()).get("snapshot_id")
            except (OSError, ValueError):
                pass
    config = {key: getattr(completer, key, None) for key in
              ("model", "reasoning_effort", "reasoning", "context_window", "n_ctx", "max_tokens", "max_output_tokens", "temperature", "top_p")}
    record(audit_log, "model_request", request_id=request_id, role=role,
           snapshot_id=snapshot_id, messages=messages, tools=getattr(completer, "tools", None), config=config)
    if completer is not None and hasattr(completer, "async_completion"):
        kwargs["completer"] = _AuditedCompleter(completer, audit_log, role, request_id)
    try:
        result = await request(**kwargs)
    except BaseException as exc:
        record(audit_log, "model_error", request_id=request_id, role=role,
               error_type=type(exc).__name__, elapsed_seconds=time.monotonic()-start)
        raise
    record(audit_log, "model_response", request_id=request_id, role=role,
           elapsed_seconds=time.monotonic()-start, response=serial(result))
    return result

def snapshot(store, root, phase, tool, details):
    """Content-addressed regular-file snapshots with manifests and readable diffs."""
    log = store.tool_log
    try:
        audit = store.run_dir / "audit"
        objects = audit / "objects"
        objects.mkdir(parents=True, exist_ok=True)
        manifest = {}
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError("submission root unavailable")
        excluded = {".git", ".venv", "venv", "__pycache__", "node_modules", ".cache"}
        omitted = []
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(p in excluded for p in relative.parts) or path.is_symlink() or not path.is_file():
                continue
            if path.name == ".env" or path.name.startswith(".env."):
                omitted.append(str(relative))
                continue
            if path.stat().st_size > 2 * 1024 * 1024:
                hasher = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
                manifest[str(relative)] = {"sha256": digest, "size": path.stat().st_size, "content_omitted": "over_2MiB"}
                continue
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            # Preserve source text after credential redaction, not environment secrets.
            try:
                text = redact_sensitive(raw.decode("utf-8"))
            except UnicodeDecodeError:
                manifest[str(relative)] = {"sha256": digest, "size": len(raw), "binary": True}
                continue
            saved = text.encode("utf-8")
            object_id = hashlib.sha256(saved).hexdigest()
            target = objects / object_id
            if not target.exists():
                target.write_bytes(saved)
            manifest[str(relative)] = {"sha256": digest, "size": len(raw), "object": object_id}
        previous_file = audit / "latest.json"
        previous = json.loads(previous_file.read_text()) if previous_file.exists() else {}
        changes = []
        for name in sorted(set(previous.get("files", {})) | set(manifest)):
            old = previous.get("files", {}).get(name, {})
            new = manifest.get(name, {})
            if old == new:
                continue
            def text(entry):
                return (objects / entry["object"]).read_text() if entry.get("object") else ""
            changes.append({"path": name, "before": old, "after": new,
                "diff": "".join(difflib.unified_diff(text(old).splitlines(True),
                       text(new).splitlines(True), fromfile="before/"+name, tofile="after/"+name))})
        snapshot_id = uuid.uuid4().hex
        value = {"snapshot_id": snapshot_id, "previous_snapshot_id": previous.get("snapshot_id"),
                 "phase": phase, "tool": tool, "details": serial(details),
                 "files": manifest, "omitted": omitted, "changes": changes}
        atomic_json_write(audit / "snapshots" / (snapshot_id+".json"), redact_sensitive(value))
        atomic_json_write(previous_file, {"snapshot_id": snapshot_id, "review_id": details.get("review_id"), "files": manifest})
        record(log, "code_snapshot", snapshot_id=snapshot_id, phase=phase, tool=tool,
               details=details, changed_files=[x["path"] for x in changes])
    except Exception as exc:
        record(log, "snapshot_error", phase=phase, tool=tool, error_type=type(exc).__name__)

