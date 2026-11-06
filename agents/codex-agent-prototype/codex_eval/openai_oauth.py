#!/usr/bin/env python3
"""Independent ChatGPT-plan OAuth login and Responses smoke probe (no agent runs)."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import fcntl
import hashlib
import hmac
from http.client import IncompleteRead, RemoteDisconnected
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import ProxyHandler, Request, build_opener
import uuid

import jwt

ROOT = Path(__file__).resolve().parents[2]
ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPES = {"resource.invoke", "chatgpt.tokens.use.direct"}
SECRET_PATTERN = re.compile(r"(?:sk-[\w.-]+|eyJ[\w-]+\.[\w-]+\.[\w-]+)")


def default_credentials_dir() -> Path:
    override = os.getenv("CODEX_GRADER_OPENAI_OAUTH_DIR") or os.getenv("OPENAI_OAUTH_DIR")
    if override:
        return Path(override).expanduser()
    config_root = Path(os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config")
    return config_root / "ppblab" / "openai-oauth"


class OAuthError(RuntimeError):
    """Safe diagnostic; never include tokens, codes, or raw token responses."""


class OAuthHTTPError(OAuthError):
    def __init__(self, message, *, status_code, code, headers=None, body=None, request_id=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.headers = headers or {}
        self.body = body
        self.request_id = request_id


def private_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OAuthError(f"Expected a JSON object in {path.name}")
    return value


def require_plan_access(record: dict) -> None:
    if not PLAN_SCOPES.issubset(set(record.get("scopes", []))):
        raise OAuthError("ChatGPT plan API permission was not granted; authorize the requested scopes.")
    if not record.get("access_token"):
        raise OAuthError("No access token is saved; run login first.")


def consume_response(stream) -> dict:
    """SSE success requires a terminal response.completed, not a text delta."""
    parts = []
    data_lines = []
    completed = None

    def event(raw):
        nonlocal completed
        if not raw or raw == "[DONE]":
            return
        value = json.loads(raw)
        kind = value.get("type")
        if kind == "response.output_text.delta":
            parts.append(str(value.get("delta", "")))
        elif kind == "response.completed":
            completed = value.get("response") or {}
        elif kind in {"response.failed", "response.incomplete", "error"}:
            response = value.get("response") or {}
            error = response.get("error") or value.get("error") or value
            code = error.get("code", kind) if isinstance(error, dict) else kind
            raise OAuthError(f"Responses stream failed: {code}")

    for raw in stream:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                event("\n".join(data_lines))
                data_lines.clear()
            if completed is not None:
                break
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    if data_lines:
        event("\n".join(data_lines))
    if completed is None:
        raise OAuthError("Stream ended without response.completed.")
    if not parts:
        for item in completed.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parts.append(content.get("text", ""))
    return {"completed": True, "model": completed.get("model"),
            "text": "".join(parts), "usage": completed.get("usage")}


class OAuthClient:
    def __init__(self, directory: Path, proxy: str | None = None):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self.credentials = self.directory / "credentials.json"
        self.status = self.directory / "login-status.json"
        self.secret_values = []
        if proxy:
            parsed = urlparse(proxy)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise OAuthError("Proxy must be HTTP(S), without embedded credentials.")
        self.opener = build_opener(ProxyHandler({"http": proxy, "https": proxy}) if proxy else ProxyHandler())
        self.discovery = None

    def safe(self, message: object) -> str:
        text = str(message)
        for secret in self.secret_values:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return SECRET_PATTERN.sub("[REDACTED]", text)[:1000]

    def open(self, request: Request, timeout=45):
        request.add_header("User-Agent", "ppblab-research-agent-oauth/0.1")
        try:
            return self.opener.open(request, timeout=timeout)
        except HTTPError as exc:
            try:
                value = json.loads(exc.read(65536))
            except (ValueError, OSError):
                value = {}
            error = value.get("error", {}) if isinstance(value, dict) else {}
            code = error.get("code", error.get("type", "unknown")) if isinstance(error, dict) else error
            request_id = exc.headers.get("x-request-id", "unavailable")
            raise OAuthHTTPError(self.safe(f"HTTP {exc.code}; error={code}; request_id={request_id}"),
                                 status_code=exc.code, code=code, body=value, request_id=request_id,
                                 headers={key: exc.headers[key] for key in ("retry-after", "retry-after-ms", "x-should-retry", "x-request-id")
                                          if key in exc.headers}) from None
        except (URLError, TimeoutError) as exc:
            raise OAuthError(self.safe(f"Network error: {exc}")) from None

    def json_request(self, url, *, data=None, headers=None) -> dict:
        try:
            with self.open(Request(url, data=data, headers=headers or {})) as response:
                value = json.load(response)
        except (IncompleteRead, RemoteDisconnected, TimeoutError, ConnectionError) as exc:
            raise OAuthError(f"Network error: OAuth response interrupted ({type(exc).__name__})") from None
        if not isinstance(value, dict):
            raise OAuthError("Server did not return a JSON object.")
        return value

    def discover(self):
        if self.discovery is None:
            value = self.json_request(ISSUER + "/.well-known/openid-configuration")
            if value.get("issuer") != ISSUER:
                raise OAuthError("Unexpected OpenAI issuer in discovery metadata.")
            for key in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
                parsed = urlparse(value.get(key, ""))
                if parsed.scheme != "https" or parsed.hostname != "auth.openai.com":
                    raise OAuthError(f"Unexpected OpenAI {key}.")
            self.discovery = value
        return self.discovery

    def verify_identity(self, token, client_id, nonce):
        discovery = self.discover()
        header = jwt.get_unverified_header(token)
        if header.get("alg") != "RS256":
            raise OAuthError("Unsupported ID token signing algorithm.")
        keys = self.json_request(discovery["jwks_uri"]).get("keys", [])
        key = next((item for item in keys if item.get("kid") == header.get("kid")), None)
        if key is None:
            raise OAuthError("ID token signing key was not found in OpenAI JWKS.")
        try:
            claims = jwt.decode(token, jwt.PyJWK.from_dict(key).key, algorithms=["RS256"],
                                issuer=ISSUER, audience=client_id, leeway=5,
                                options={"require": ["sub", "exp", "iat", "iss", "aud"]})
        except jwt.PyJWTError as exc:
            raise OAuthError(f"ID token verification failed: {type(exc).__name__}") from None
        if not isinstance(claims.get("sub"), str) or not claims["sub"]:
            raise OAuthError("ID token subject is missing.")
        if not hmac.compare_digest(str(claims.get("nonce", "")), nonce):
            raise OAuthError("ID token nonce does not match the authorization attempt.")
        return claims

    def token_exchange(self, values):
        for key in ("code", "code_verifier", "refresh_token"):
            self.secret_values.append(values.get(key, ""))
        value = self.json_request(self.discover()["token_endpoint"],
                                  data=urlencode(values).encode(),
                                  headers={"Content-Type": "application/x-www-form-urlencoded"})
        self.secret_values.extend(value.get(key, "") for key in ("access_token", "refresh_token", "id_token"))
        return value

    def save_tokens(self, response, *, previous, identity=None, client_id=None, host_id=None):
        record = dict(previous)
        if identity is not None:
            if previous.get("subject") and identity["sub"] != previous["subject"]:
                raise OAuthError("Authorized account differs from the saved account; credentials were not replaced.")
            record.update(subject=identity["sub"], email=identity.get("email"), issuer=ISSUER,
                          client_id=client_id, ext_agent_host_id=host_id)
        if not response.get("access_token") or not response.get("refresh_token"):
            raise OAuthError("Token response lacks access_token or rotating refresh_token.")
        if str(response.get("token_type", "")).lower() != "bearer":
            raise OAuthError("Unexpected OAuth token type.")
        now = time.time()
        record.update(access_token=response["access_token"], refresh_token=response["refresh_token"],
                      token_type="Bearer", expires_in=int(response.get("expires_in", 3600)), saved_at=now,
                      expires_at=now + int(response.get("expires_in", 3600)),
                      earliest_refresh_at=response.get("earliest_refresh_at"))
        if response.get("id_token"):
            record["id_token"] = response["id_token"]
        if "scope" in response:
            record["scopes"] = str(response["scope"]).split()
        elif identity is not None:
            raise OAuthError("Authorization response omitted granted scopes.")
        record.pop("reauthorization_required", None)
        private_json(self.credentials, record)
        return record

    def refresh(self, *, force=False, failed_access_token=None):
        lock_path = self.directory / "refresh.lock"
        with os.fdopen(os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            record = read_json(self.credentials)
            require_plan_access(record)
            if failed_access_token and record["access_token"] != failed_access_token:
                return record
            if not force and record.get("expires_at", 0) > time.time() + 300:
                return record
            earliest = record.get("earliest_refresh_at")
            if isinstance(earliest, str):
                try:
                    earliest = float(earliest)
                except ValueError:
                    earliest = datetime.fromisoformat(earliest.replace("Z", "+00:00")).timestamp()
            if isinstance(earliest, (int, float)) and earliest > time.time():
                if not force and record.get("expires_at", 0) > time.time():
                    return record
                raise OAuthError("The server's earliest_refresh_at has not been reached.")
            try:
                response = self.token_exchange({"grant_type": "refresh_token", "client_id": record["client_id"],
                                               "refresh_token": record["refresh_token"], "resource": RESOURCE})
            except OAuthHTTPError as exc:
                if exc.code in {"invalid_grant", "invalid_refresh_token", "token_expired", "refresh_token_expired",
                                "refresh_token_invalidated", "refresh_token_reused"}:
                    # Keep account/client/host identity for reauthorization, but
                    # never keep using tokens the server explicitly invalidated.
                    for key in ("access_token", "refresh_token", "expires_at", "earliest_refresh_at"):
                        record.pop(key, None)
                    record["reauthorization_required"] = True
                    private_json(self.credentials, record)
                raise
            updated = self.save_tokens(response, previous=record)
            require_plan_access(updated)
            return updated

    def models(self):
        record = self.refresh()
        self.secret_values.extend([record["access_token"], record["refresh_token"]])
        data = self.json_request(RESOURCE + "/models", headers={"Authorization": "Bearer " + record["access_token"]})
        models = data.get("models", data.get("data", []))
        if not isinstance(models, list):
            raise OAuthError("Model catalog is not an array.")
        visible = [{"slug": m.get("slug", m.get("id")), "display_name": m.get("display_name", m.get("id"))}
                   for m in models if isinstance(m, dict) and m.get("visibility", "list") == "list"]
        private_json(self.directory / "models.json", {"models": visible, "fetched_at": time.time()})
        return visible

    def probe(self, model):
        if model not in {m["slug"] for m in self.models()}:
            raise OAuthError(f"Requested model {model} is not in this account's visible model catalog; no substitute was used.")
        record = self.refresh()
        body = {"model": model, "instructions": "Respond with exactly OK.",
                "input": [{"role": "user", "content": "Respond with exactly OK."}],
                "store": False, "stream": True}
        with self.open(Request(RESOURCE + "/responses", data=json.dumps(body).encode(),
                               headers={"Authorization": "Bearer " + record["access_token"],
                                        "Content-Type": "application/json", "Accept": "text/event-stream"}), timeout=120) as stream:
            result = consume_response(stream)
        private_json(self.directory / "probe-result.json", result)
        print(json.dumps({"probe": "completed", "requested_model": model, "response": result["text"]}, ensure_ascii=False), flush=True)
        return result

    def login(self, *, port, timeout, probe_model=None):
        host_path = self.directory / "host.json"
        host = read_json(host_path)
        if not host.get("ext_agent_host_id"):
            host = {"ext_agent_host_id": "urn:uuid:" + str(uuid.uuid4())}
            private_json(host_path, host)
        previous = read_json(self.credentials)
        registration = read_json(self.directory / "registration.json")
        client_id = previous.get("client_id") or registration.get("client_id") or "dynamic_agent_client"
        state, nonce, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(32), secrets.token_urlsafe(64)
        self.secret_values.append(verifier)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        outcome = {}
        client = self

        class Callback(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass  # Callback URLs contain authorization codes.

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path != "/auth/callback":
                    self.send_error(404)
                    return
                query = parse_qs(parsed.query)
                returned_state = query.get("state", [""])[0]
                if not hmac.compare_digest(returned_state, state):
                    self.send_error(400, "OAuth state does not match")
                    return
                try:
                    if query.get("error"):
                        raise OAuthError("Authorization declined or failed: " + query["error"][0])
                    issued = query.get("client_id", [client_id])[0]
                    if issued == "dynamic_agent_client" or (client_id != "dynamic_agent_client" and issued != client_id):
                        raise OAuthError("Callback client ID is missing or does not match.")
                    code = query.get("code", [""])[0]
                    if not code:
                        raise OAuthError("Callback contains no authorization code.")
                    private_json(client.directory / "registration.json", {"client_id": issued})
                    response = client.token_exchange({"grant_type": "authorization_code", "client_id": issued,
                        "code": code, "code_verifier": verifier, "redirect_uri": redirect_uri, "resource": RESOURCE})
                    if not response.get("id_token"):
                        raise OAuthError("Authorization response lacks id_token.")
                    identity = client.verify_identity(response["id_token"], issued, nonce)
                    record = client.save_tokens(response, previous=previous, identity=identity,
                                                client_id=issued, host_id=host["ext_agent_host_id"])
                    require_plan_access(record)
                    outcome["ok"] = True
                    private_json(client.status, {"phase": "authorized", "saved_at": time.time()})
                    message = "授权成功，凭据已安全保存。脚本将继续进行最小 API 测试，可关闭此页面。"
                    self.send_response(200)
                except Exception as exc:
                    outcome["error"] = client.safe(exc)
                    private_json(client.status, {"phase": "failed", "error": outcome["error"]})
                    message = "授权未完成。请返回终端查看诊断信息。"
                    self.send_response(400)
                data = ("<!doctype html><meta charset=utf-8><title>ResearchAgent OAuth</title><p>" + message + "</p>").encode()
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

        with HTTPServer(("127.0.0.1", port), Callback) as server:
            server.timeout = 1
            redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
            params = {"client_id": client_id, "ext_agent_host_id": host["ext_agent_host_id"],
                      "response_type": "code", "redirect_uri": redirect_uri, "scope": SCOPES,
                      "resource": RESOURCE, "state": state, "nonce": nonce,
                      "code_challenge_method": "S256", "code_challenge": challenge}
            if client_id == "dynamic_agent_client":
                params["agent_name_hint"] = "PPLab ResearchAgent"
            url = self.discover()["authorization_endpoint"] + "?" + urlencode(params)
            private_json(self.status, {"phase": "listening", "callback": redirect_uri,
                                      "expires_at": time.time() + timeout})
            print("Continue with ChatGPT:\n" + url, flush=True)
            deadline = time.monotonic() + timeout
            while not outcome and time.monotonic() < deadline:
                server.handle_request()
        if not outcome:
            raise OAuthError("Authorization callback timed out; run login again for a new link.")
        if outcome.get("error"):
            raise OAuthError(outcome["error"])
        print("OAuth identity and ChatGPT-plan scopes verified; credentials saved with mode 0600.", flush=True)
        if probe_model:
            self.probe(probe_model)
            private_json(self.status, {"phase": "probe_completed", "model": probe_model,
                                      "completed_at": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-dir", type=Path, default=default_credentials_dir())
    parser.add_argument("--proxy", help="Optional HTTP(S) proxy without credentials")
    modes = parser.add_subparsers(dest="mode", required=True)
    login = modes.add_parser("login", help="Authorize this RA client in a browser")
    login.add_argument("--port", type=int, default=1455)
    login.add_argument("--timeout", type=int, default=1800)
    login.add_argument("--probe-model", default="gpt-6-luna")
    modes.add_parser("models", help="List models available to the authorized account")
    probe = modes.add_parser("probe", help="Run one tiny streaming inference request")
    probe.add_argument("--model", default="gpt-6-luna")
    modes.add_parser("refresh", help="Exercise token rotation (respect server refresh timing)")
    modes.add_parser("status", help="Show non-secret status only")
    args = parser.parse_args()
    client = OAuthClient(args.credentials_dir, args.proxy)
    try:
        if args.mode == "login":
            client.login(port=args.port, timeout=args.timeout, probe_model=args.probe_model)
        elif args.mode == "models":
            print(json.dumps(client.models(), ensure_ascii=False, indent=2))
        elif args.mode == "probe":
            client.probe(args.model)
        elif args.mode == "refresh":
            client.refresh(force=True)
            print("OAuth refresh completed; replacement credentials saved.")
        else:
            record = read_json(client.credentials)
            print(json.dumps({"status": read_json(client.status), "authenticated": bool(record),
                              "plan_access": PLAN_SCOPES.issubset(set(record.get("scopes", []))),
                              "expires_at": record.get("expires_at")}, ensure_ascii=False, indent=2))
    except Exception as exc:
        error = client.safe(exc)
        private_json(client.status, {"phase": "failed", "error": error, "updated_at": time.time()})
        print(error, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
