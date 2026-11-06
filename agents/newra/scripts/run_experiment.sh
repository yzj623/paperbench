#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 PAPER_ID" >&2
  exit 2
fi

PAPER_ID="$1"
RESEARCH_AGENT_ROOT="${RESEARCH_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "$RESEARCH_AGENT_ROOT/.env" ]]; then
  set -a
  source "$RESEARCH_AGENT_ROOT/.env"
  set +a
fi
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$(cd "$RESEARCH_AGENT_ROOT/../.." && pwd)/experiments}"
STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
RUN_ROOT="${RESEARCH_AGENT_RUN_ROOT:-$EXPERIMENTS_ROOT/$PAPER_ID/$STAMP}"

"$RESEARCH_AGENT_ROOT/scripts/run_agent.sh" "$PAPER_ID" "$RUN_ROOT"
SUBMISSION_DIR="$(cat "$RUN_ROOT/FINAL_SUBMISSION_PATH")"
"$RESEARCH_AGENT_ROOT/scripts/evaluate_submission.sh" \
  "$PAPER_ID" "$SUBMISSION_DIR" "$RUN_ROOT/evaluation"

echo "End-to-end experiment complete: $RUN_ROOT"

