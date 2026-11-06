#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN="$ROOT/plans/all-in-one.plan.json"
EXPECTED_PLAN_SHA256=99a0560b4ae9846ce81915bc54b31e293fe5d6ee0129e086d51b704e8747c68d
[[ -f "$PLAN" ]] || { echo "Missing fixed outline: $PLAN" >&2; exit 1; }
[[ "$(sha256sum "$PLAN" | cut -d' ' -f1)" == "$EXPECTED_PLAN_SHA256" ]] || {
  echo "Fixed outline does not match the ordinary RA all-in-one outline" >&2
  exit 1
}
[[ -f "${RESEARCH_AGENT_OPENAI_OAUTH_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/ppblab/openai-oauth}/credentials.json" ]] || {
  echo "OpenAI OAuth credentials are missing" >&2
  exit 1
}

export RESEARCH_AGENT_ROOT="$ROOT"
export EXPERIMENTS_ROOT="$ROOT/runs"
export RESEARCH_AGENT_PLAN="$PLAN"
export RESEARCH_AGENT_API_CONFIG=openai-oauth
export RESEARCH_AGENT_MODEL=gpt-6-luna
export RESEARCH_AGENT_REASONING_EFFORT=medium
export RESEARCH_AGENT_REVIEW_ROUNDS=2
export PAPERBENCH_GRADER_API_CONFIG=ustc-deepseek-flash

STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
RUN_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
export RESEARCH_AGENT_RUN_ROOT="$EXPERIMENTS_ROOT/all-in-one/${STAMP}-gpt6luna-review2-${RUN_ID}"
mkdir -p "$RESEARCH_AGENT_RUN_ROOT"
printf '%s\n' "$RESEARCH_AGENT_RUN_ROOT" > "$EXPERIMENTS_ROOT/LAST_NEWRA_RUN"
echo "Run root: $RESEARCH_AGENT_RUN_ROOT"
echo "Agent: gpt-6-luna, reasoning medium; full review rounds: 2"
echo "Fixed plan SHA256: $EXPECTED_PLAN_SHA256"
echo "Grader: USTC deepseek-flash-2"

"$ROOT/scripts/run_experiment.sh" all-in-one
