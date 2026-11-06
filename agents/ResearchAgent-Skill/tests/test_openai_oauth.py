"""Offline checks: no real credentials or API requests."""
import io
import json
import stat
import time

import pytest

from research_agent import openai_oauth as oauth


def test_shared_store_default_and_override(monkeypatch, tmp_path):
    from research_agent.openai_oauth_completer import default_credentials_dir
    monkeypatch.delenv("RESEARCH_AGENT_OPENAI_OAUTH_DIR", raising=False)
    monkeypatch.delenv("OPENAI_OAUTH_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    expected = tmp_path / "config" / "ppblab" / "openai-oauth"
    assert oauth.default_credentials_dir() == expected
    assert default_credentials_dir() == str(expected)
    monkeypatch.setenv("OPENAI_OAUTH_DIR", str(tmp_path / "shared"))
    assert oauth.default_credentials_dir() == tmp_path / "shared"


def test_private_atomic_save(tmp_path):
    path = tmp_path / "private" / "credentials.json"
    oauth.private_json(path, {"value": 1})
    oauth.private_json(path, {"value": 2})
    assert oauth.read_json(path) == {"value": 2}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_plan_permission_required():
    with pytest.raises(oauth.OAuthError):
        oauth.require_plan_access({"access_token": "dummy", "scopes": ["openid"]})


def sse(*events):
    return io.BytesIO("".join("data: " + json.dumps(e) + "\n\n" for e in events).encode())


def test_stream_requires_completion():
    delta = {"type": "response.output_text.delta", "delta": "OK"}
    with pytest.raises(oauth.OAuthError, match="without response.completed"):
        oauth.consume_response(sse(delta))
    result = oauth.consume_response(sse(delta, {"type": "response.completed", "response": {"model": "test"}}))
    assert result["completed"] and result["text"] == "OK"


@pytest.mark.parametrize("kind", ["response.failed", "response.incomplete", "error"])
def test_stream_failure(kind):
    with pytest.raises(oauth.OAuthError):
        oauth.consume_response(sse({"type": kind}))


def test_refresh_rotation(tmp_path, monkeypatch):
    client = oauth.OAuthClient(tmp_path / "credentials")
    original = {"client_id": "test-client", "access_token": "old-access",
                "refresh_token": "old-refresh", "scopes": sorted(oauth.PLAN_SCOPES), "expires_at": 0}
    oauth.private_json(client.credentials, original)
    calls = []

    def exchange(values):
        calls.append(values)
        return {"access_token": "new-access", "refresh_token": "new-refresh", "token_type": "Bearer", "expires_in": 3600}

    monkeypatch.setattr(client, "token_exchange", exchange)
    updated = client.refresh()
    assert updated["refresh_token"] == "new-refresh"
    assert "scope" not in calls[0]
    assert calls[0]["resource"] == oauth.RESOURCE
    assert oauth.read_json(client.credentials) == updated
    assert client.refresh() == updated
    assert len(calls) == 1


def test_refresh_failure_preserves_credentials(tmp_path, monkeypatch):
    client = oauth.OAuthClient(tmp_path / "credentials")
    original = {"client_id": "test-client", "access_token": "old-access", "refresh_token": "old-refresh",
                "scopes": sorted(oauth.PLAN_SCOPES), "expires_at": 0}
    oauth.private_json(client.credentials, original)

    def failure(_):
        raise oauth.OAuthError("Network error")

    monkeypatch.setattr(client, "token_exchange", failure)
    with pytest.raises(oauth.OAuthError):
        client.refresh()
    assert oauth.read_json(client.credentials) == original


def test_concurrent_rotation_and_server_timing(tmp_path, monkeypatch):
    client = oauth.OAuthClient(tmp_path / "credentials")
    record = {"client_id": "test", "access_token": "rotated", "refresh_token": "refresh",
              "scopes": sorted(oauth.PLAN_SCOPES), "expires_at": time.time() + 60,
              "earliest_refresh_at": time.time() + 30}
    oauth.private_json(client.credentials, record)

    def unexpected(_):
        pytest.fail("No token exchange should occur")

    monkeypatch.setattr(client, "token_exchange", unexpected)
    assert client.refresh(force=True, failed_access_token="obsolete") == record
    assert client.refresh() == record


def test_parallel_refresh_is_serialized(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    directory = tmp_path / "credentials"
    original = {"client_id": "test", "access_token": "old", "refresh_token": "old-refresh",
                "scopes": sorted(oauth.PLAN_SCOPES), "expires_at": 0}
    oauth.private_json(directory / "credentials.json", original)
    calls, lock = [], Lock()

    def exchange(self, values):
        with lock:
            calls.append(values)
        time.sleep(0.02)
        return {"access_token": "new", "refresh_token": "new-refresh", "token_type": "Bearer", "expires_in": 3600}

    monkeypatch.setattr(oauth.OAuthClient, "token_exchange", exchange)
    clients = [oauth.OAuthClient(directory) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda client: client.refresh(), clients))
    assert len(calls) == 1
    assert all(item["refresh_token"] == "new-refresh" for item in results)


def test_nonce_and_signature(tmp_path, monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import rsa

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(oauth.jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    public_jwk["kid"] = "test-key"
    client = oauth.OAuthClient(tmp_path / "credentials")
    monkeypatch.setattr(client, "discover", lambda: {"jwks_uri": "https://auth.openai.com/test"})
    monkeypatch.setattr(client, "json_request", lambda _: {"keys": [public_jwk]})
    token = oauth.jwt.encode({"iss": oauth.ISSUER, "aud": "client", "sub": "user",
                             "iat": int(time.time()), "exp": int(time.time()) + 60, "nonce": "expected"},
                            private, algorithm="RS256", headers={"kid": "test-key"})
    assert client.verify_identity(token, "client", "expected")["sub"] == "user"
    with pytest.raises(oauth.OAuthError, match="nonce"):
        client.verify_identity(token, "client", "wrong")
    with pytest.raises(oauth.OAuthError, match="verification failed"):
        client.verify_identity(token, "another-client", "expected")
