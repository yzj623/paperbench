#!/usr/bin/env bash
set -euo pipefail

MONITOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$(cd "$MONITOR_ROOT/.." && pwd)/experiments}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8765}"

[[ -d "$EXPERIMENTS_ROOT" ]] || {
  echo "Experiments root does not exist: $EXPERIMENTS_ROOT" >&2
  exit 1
}

PYTHON_BIN="$(command -v python3 || true)"
[[ -n "$PYTHON_BIN" ]] || {
  echo "Python 3 was not found." >&2
  exit 1
}

mkdir -p "$MONITOR_ROOT/logs" "$MONITOR_ROOT/cache"
export EXPERIMENTS_ROOT WEB_HOST WEB_PORT
exec "$PYTHON_BIN" "$MONITOR_ROOT/server.py" \
  >>"$MONITOR_ROOT/logs/stdout.log" 2>&1
