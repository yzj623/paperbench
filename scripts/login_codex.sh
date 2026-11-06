#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--browser" && "$1" != "--status" ) ]]; then
  echo "Usage: $0 [--browser|--status]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AUTH_DIR="${CODEX_AUTH_DIR:-$REPO_ROOT/agents/codex-agent-prototype/.codex-auth}"

umask 077
mkdir -p "$AUTH_DIR"
AUTH_DIR="$(cd "$AUTH_DIR" && pwd)"
chmod 700 "$AUTH_DIR"
export CODEX_HOME="$AUTH_DIR"

if [[ -n "${CODEX_CLI:-}" ]]; then
  [[ -x "$CODEX_CLI" ]] || { echo "CODEX_CLI is not executable: $CODEX_CLI" >&2; exit 1; }
  cli=("$CODEX_CLI")
elif command -v codex >/dev/null 2>&1; then
  cli=(codex)
elif command -v npx >/dev/null 2>&1; then
  cli=(npx --yes @openai/codex)
else
  echo "Install Codex CLI or Node.js/npx, then rerun this script." >&2
  exit 1
fi

config='cli_auth_credentials_store="file"'
if [[ "${1:-}" == "--status" ]]; then
  exec "${cli[@]}" login -c "$config" status
fi

echo "Codex credentials will be saved under: $AUTH_DIR"
if [[ "${1:-}" == "--browser" ]]; then
  "${cli[@]}" login -c "$config"
else
  "${cli[@]}" login -c "$config" --device-auth
fi

[[ -s "$AUTH_DIR/auth.json" ]] || {
  echo "Login ended without $AUTH_DIR/auth.json. Check the login output and retry." >&2
  exit 1
}
chmod 600 "$AUTH_DIR/auth.json"
"${cli[@]}" login -c "$config" status
echo "Codex login is ready for agents/codex-agent-prototype."
