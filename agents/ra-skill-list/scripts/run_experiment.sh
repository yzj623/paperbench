#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 PAPER_ID PLAN_JSON [tools|embedded]" >&2
  echo "PLAN_JSON may be a legacy outline without checklists or a complete checklist plan." >&2
  exit 2
fi

PAPER_ID="$1"
PLAN_INPUT="$2"
CHECKER_MATERIALS_MODE="${3:-${RESEARCH_AGENT_CHECKER_MATERIALS:-tools}}"
case "$CHECKER_MATERIALS_MODE" in
  tools|embedded) ;;
  *) echo "Invalid checker materials mode: $CHECKER_MATERIALS_MODE (expected tools or embedded)" >&2; exit 2 ;;
esac
if [[ ! "$PAPER_ID" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "Invalid PaperBench paper ID: $PAPER_ID" >&2
  exit 2
fi
[[ -f "$PLAN_INPUT" ]] || {
  echo "Plan does not exist: $PLAN_INPUT" >&2
  exit 2
}
PLAN_FILE="$(realpath -e "$PLAN_INPUT")"
RESEARCH_AGENT_ROOT="${RESEARCH_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="$RESEARCH_AGENT_ROOT/.venv-wsl/bin/python"
[[ -x "$PYTHON" ]] || {
  echo "Missing local Python environment. Run: $RESEARCH_AGENT_ROOT/scripts/setup.sh" >&2
  exit 1
}
GRADER_API_OVERRIDE="${PAPERBENCH_GRADER_API_CONFIG:-}"
SKILLS_OVERRIDE="${RESEARCH_AGENT_ENABLE_SKILLS:-}"
if [[ -f "$RESEARCH_AGENT_ROOT/.env" ]]; then
  set -a
  source "$RESEARCH_AGENT_ROOT/.env"
  set +a
fi
if [[ -n "$GRADER_API_OVERRIDE" ]]; then
  export PAPERBENCH_GRADER_API_CONFIG="$GRADER_API_OVERRIDE"
fi
if [[ -n "$SKILLS_OVERRIDE" ]]; then
  export RESEARCH_AGENT_ENABLE_SKILLS="$SKILLS_OVERRIDE"
fi
PLAN_KIND="$(PYTHONPATH="$RESEARCH_AGENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -c '
import json
import sys
from pathlib import Path
from research_agent.plan import parse_plan
from research_agent.plan_enrichment import is_legacy_outline, parse_legacy_outline

path = Path(sys.argv[1])
try:
    value = json.loads(path.read_text(encoding="utf-8"))
    if is_legacy_outline(value):
        parse_legacy_outline(value)
        print("legacy")
    else:
        parse_plan(value)
        print("complete")
except (OSError, UnicodeError, ValueError) as exc:
    print(f"Invalid plan {path}: {exc}", file=sys.stderr)
    sys.exit(2)
' "$PLAN_FILE")"

case "${PAPERBENCH_GRADER_API_CONFIG:-ustc-deepseek-flash}" in
  ustc-deepseek-flash)
    [[ -n "${USTC_LLM_API_KEY:-}" ]] || {
      echo "USTC_LLM_API_KEY is required for the grader." >&2
      exit 1
    }
    ;;
  deepseek-official-flash)
    [[ -n "${DEEPSEEK_API_KEY:-}" ]] || {
      echo "DEEPSEEK_API_KEY is required for the grader." >&2
      exit 1
    }
    ;;
  openai)
    [[ -n "${GRADER_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}" ]] || {
      echo "GRADER_OPENAI_API_KEY or OPENAI_API_KEY is required for the grader." >&2
      exit 1
    }
    ;;
  *)
    echo "Unknown PAPERBENCH_GRADER_API_CONFIG: ${PAPERBENCH_GRADER_API_CONFIG}" >&2
    exit 2
    ;;
esac

EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$(cd "$RESEARCH_AGENT_ROOT/../.." && pwd)/experiments}"
STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
mkdir -p "$EXPERIMENTS_ROOT/$PAPER_ID"
RUN_ROOT="$(mktemp -d "$EXPERIMENTS_ROOT/$PAPER_ID/$STAMP-ra-skill-list.XXXXXX")"
cp "$PLAN_FILE" "$RUN_ROOT/input_plan.json"
if [[ "$PLAN_KIND" == legacy ]]; then
  cp "$PLAN_FILE" "$RUN_ROOT/input_outline.json"
fi

RESEARCH_AGENT_CHECKER_MATERIALS="$CHECKER_MATERIALS_MODE" \
  "$RESEARCH_AGENT_ROOT/scripts/run_agent.sh" "$PAPER_ID" "$RUN_ROOT" "$RUN_ROOT/input_plan.json"
SUBMISSION_DIR="$(cat "$RUN_ROOT/FINAL_SUBMISSION_PATH")"
"$RESEARCH_AGENT_ROOT/scripts/evaluate_submission.sh" \
  "$PAPER_ID" "$SUBMISSION_DIR" "$RUN_ROOT/evaluation"

echo "End-to-end experiment complete: $RUN_ROOT"
