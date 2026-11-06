#!/usr/bin/env bash
# No model requests, credentials, login, Docker runs, or PaperBench grading.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
"$PROJECT_ROOT/.venv/bin/python" -m unittest discover -s tests -v
"$PROJECT_ROOT/.venv/bin/python" -m unittest discover -s monitor/tests -v
node --check "$PROJECT_ROOT/monitor/static/app.js"
bash -n "$PROJECT_ROOT/monitor/start_monitor.sh"
for script in "$PROJECT_ROOT"/*.sh "$PROJECT_ROOT"/scripts/*.sh; do
  bash -n "$script"
done
git -C "$PROJECT_ROOT/codex" diff --check
