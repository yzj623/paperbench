#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 AGENT PAPER_ID [agent-specific arguments]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AGENT="$1"
PAPER_ID="$2"
shift 2
AGENT_ROOT="$REPO_ROOT/agents/$AGENT"
[[ -d "$AGENT_ROOT" ]] || { echo "Unknown agent: $AGENT" >&2; exit 2; }

"$SCRIPT_DIR/prepare_paper.sh" "$PAPER_ID"

if [[ "$AGENT" == codex-agent-prototype ]]; then
  PAPERBENCH_ROOT="$AGENT_ROOT/vendor/paperbench-official" \
    "$SCRIPT_DIR/prepare_paper.sh" "$PAPER_ID"
  exec "$AGENT_ROOT/.venv/bin/python" "$AGENT_ROOT/run_paperbench.py" \
    --paper-id "$PAPER_ID" "$@"
fi

exec "$AGENT_ROOT/scripts/run_experiment.sh" "$PAPER_ID" "$@"
