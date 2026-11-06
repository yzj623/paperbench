#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash $0 PAPER_ID SOURCE_DIR [--plan PLAN_JSON] [--state STATE_JSON] [--checker-materials tools|embedded] [--prepare-only]" >&2
  exit 2
fi
PAPER_ID="$1"
SOURCE_DIR="$2"
shift 2
PREPARE_ONLY=false
CHECKER_MATERIALS_MODE="${RESEARCH_AGENT_CHECKER_MATERIALS:-tools}"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan|--state)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --prepare-only) PREPARE_ONLY=true; shift ;;
    --checker-materials)
      [[ $# -ge 2 ]] || { echo "Missing value for --checker-materials" >&2; exit 2; }
      CHECKER_MATERIALS_MODE="$2"
      shift 2
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$CHECKER_MATERIALS_MODE" in
  tools|embedded) ;;
  *) echo "Invalid checker materials mode: $CHECKER_MATERIALS_MODE (expected tools or embedded)" >&2; exit 2 ;;
esac
RESEARCH_AGENT_ROOT="${RESEARCH_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="$RESEARCH_AGENT_ROOT/.venv-wsl/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing Python environment: $PYTHON" >&2; exit 1; }
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
if [[ "$PREPARE_ONLY" == false ]]; then
  case "${PAPERBENCH_GRADER_API_CONFIG:-ustc-deepseek-flash}" in
    ustc-deepseek-flash) [[ -n "${USTC_LLM_API_KEY:-}" ]] || { echo "USTC_LLM_API_KEY is required" >&2; exit 1; } ;;
    deepseek-official-flash) [[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "DEEPSEEK_API_KEY is required" >&2; exit 1; } ;;
    openai) [[ -n "${GRADER_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}" ]] || { echo "Grader API key is required" >&2; exit 1; } ;;
    *) echo "Unknown grader API configuration" >&2; exit 2 ;;
  esac
fi
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$(cd "$RESEARCH_AGENT_ROOT/../.." && pwd)/experiments}"
RUN_ROOT="$(PYTHONPATH="$RESEARCH_AGENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m research_agent.resume --paper-id "$PAPER_ID" --source "$SOURCE_DIR" \
  --experiments-root "$EXPERIMENTS_ROOT" "${EXTRA_ARGS[@]}")"
echo "Continuation run root: $RUN_ROOT"
if [[ "$PREPARE_ONLY" == true ]]; then
  echo "Prepared only; agent and grader were not started."
  exit 0
fi
RESEARCH_AGENT_CHECKER_MATERIALS="$CHECKER_MATERIALS_MODE" \
  "$RESEARCH_AGENT_ROOT/scripts/run_agent.sh" "$PAPER_ID" "$RUN_ROOT" "$RUN_ROOT/input_plan.json"
SUBMISSION_DIR="$(cat "$RUN_ROOT/FINAL_SUBMISSION_PATH")"
"$RESEARCH_AGENT_ROOT/scripts/evaluate_submission.sh" "$PAPER_ID" "$SUBMISSION_DIR" "$RUN_ROOT/evaluation"
echo "Continuation and Code-Dev evaluation complete: $RUN_ROOT"
