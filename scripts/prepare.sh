#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 AGENT PAPER_ID" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AGENT="$1"
PAPER_ID="$2"
AGENT_ROOT="$REPO_ROOT/agents/$AGENT"
[[ -d "$AGENT_ROOT" ]] || { echo "Unknown agent: $AGENT" >&2; exit 2; }

"$SCRIPT_DIR/prepare_paper.sh" "$PAPER_ID"

if [[ "$AGENT" == codex-agent-prototype ]]; then
  PAPERBENCH_ROOT="$AGENT_ROOT/vendor/paperbench-official" \
    "$SCRIPT_DIR/prepare_paper.sh" "$PAPER_ID"
  "$REPO_ROOT/agents/ResearchAgent/scripts/setup.sh"
  python3 -m venv "$AGENT_ROOT/.venv"
  "$AGENT_ROOT/.venv/bin/python" -m pip install -r "$AGENT_ROOT/requirements.txt"
  "$AGENT_ROOT/build_codex_image.sh" --jobs "${BUILD_JOBS:-1}"
else
  [[ -x "$AGENT_ROOT/scripts/setup.sh" ]] || { echo "No setup script: $AGENT" >&2; exit 2; }
  "$AGENT_ROOT/scripts/setup.sh"
fi
echo "Ready to run $AGENT on $PAPER_ID"
