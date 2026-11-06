#!/usr/bin/env bash
set -euo pipefail

monitor_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export CODEX_RUNS_ROOT="${CODEX_RUNS_ROOT:-$(dirname -- "$monitor_root")/runs}"
export WEB_HOST="${WEB_HOST:-127.0.0.1}"
export WEB_PORT="${WEB_PORT:-8768}"

[[ -d "$CODEX_RUNS_ROOT" ]] || {
  echo "Codex runs root does not exist: $CODEX_RUNS_ROOT" >&2
  exit 1
}
mkdir -p "$monitor_root/logs"
exec python3 "$monitor_root/server.py" "$@" >>"$monitor_root/logs/stdout.log" 2>&1
