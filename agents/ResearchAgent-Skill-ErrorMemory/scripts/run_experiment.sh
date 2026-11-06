#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 PAPER_ID [OUTLINE_JSON]" >&2
  exit 2
fi

PAPER_ID="$1"
OUTLINE_SOURCE="${2:-}"
if [[ ! "$PAPER_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || "$PAPER_ID" == *..* ]]; then
  echo "Invalid PaperBench paper id: $PAPER_ID" >&2
  exit 2
fi
if [[ -n "$OUTLINE_SOURCE" && ! -f "$OUTLINE_SOURCE" ]]; then
  echo "Specified outline file does not exist: $OUTLINE_SOURCE" >&2
  exit 2
fi
RESEARCH_AGENT_ROOT="${RESEARCH_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SKILLS_OVERRIDE="${RESEARCH_AGENT_ENABLE_SKILLS:-}"
if [[ -f "$RESEARCH_AGENT_ROOT/.env" ]]; then
  set -a
  source "$RESEARCH_AGENT_ROOT/.env"
  set +a
fi
if [[ -n "$SKILLS_OVERRIDE" ]]; then
  export RESEARCH_AGENT_ENABLE_SKILLS="$SKILLS_OVERRIDE"
fi
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$(cd "$RESEARCH_AGENT_ROOT/../.." && pwd)/experiments}"
PAPERBENCH_ROOT="${PAPERBENCH_ROOT:-$(cd "$RESEARCH_AGENT_ROOT/../.." && pwd)/PaperBench-Official}"
if [[ -n "$OUTLINE_SOURCE" ]]; then
  python3 - "$OUTLINE_SOURCE" <<'PY'
import json
import re
import sys

path = sys.argv[1]
try:
    plan = json.load(open(path, encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid outline JSON {path}: {exc}")
if not isinstance(plan, dict) or set(plan) != {"subtasks"}:
    raise SystemExit("Outline must be a JSON object containing only 'subtasks'.")
tasks = plan["subtasks"]
if not isinstance(tasks, list) or not tasks:
    raise SystemExit("Outline 'subtasks' must be a non-empty array.")
seen = set()
for index, task in enumerate(tasks, 1):
    if not isinstance(task, dict) or set(task) != {"id", "name", "description"}:
        raise SystemExit(f"Outline subtask {index} must contain only id, name, description.")
    task_id = task["id"]
    if not isinstance(task_id, str) or not re.fullmatch(r"task_[0-9]{3,}", task_id):
        raise SystemExit(f"Outline subtask {index} has an invalid id: {task_id!r}")
    if task_id in seen:
        raise SystemExit(f"Outline has duplicate subtask id: {task_id}")
    seen.add(task_id)
    if any(not isinstance(task[field], str) or not task[field].strip() for field in ("name", "description")):
        raise SystemExit(f"Outline subtask {index} name and description must be non-empty strings.")
PY
fi
STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
mkdir -p "$EXPERIMENTS_ROOT/$PAPER_ID"
RUN_ROOT="$(mktemp -d "$EXPERIMENTS_ROOT/$PAPER_ID/$STAMP.XXXXXX")"

if [[ -n "$OUTLINE_SOURCE" ]]; then
  cp -- "$OUTLINE_SOURCE" "$RUN_ROOT/input_outline.json"
  OUTLINE_RUN_COPY="$RUN_ROOT/input_outline.json"
else
  OUTLINE_RUN_COPY=""
fi

if [[ ! -x "$RESEARCH_AGENT_ROOT/.venv-wsl/bin/python" ]] || \
  ! docker image inspect pb-env:latest >/dev/null 2>&1; then
  PAPERBENCH_ROOT="$PAPERBENCH_ROOT" RESEARCH_AGENT_ROOT="$RESEARCH_AGENT_ROOT" \
    bash "$RESEARCH_AGENT_ROOT/scripts/setup.sh"
fi

RESEARCH_AGENT_PLAN="$OUTLINE_RUN_COPY" \
  "$RESEARCH_AGENT_ROOT/scripts/run_agent.sh" "$PAPER_ID" "$RUN_ROOT"
SUBMISSION_DIR="$(cat "$RUN_ROOT/FINAL_SUBMISSION_PATH")"
"$RESEARCH_AGENT_ROOT/scripts/evaluate_submission.sh" \
  "$PAPER_ID" "$SUBMISSION_DIR" "$RUN_ROOT/evaluation"

echo "End-to-end experiment complete: $RUN_ROOT"
