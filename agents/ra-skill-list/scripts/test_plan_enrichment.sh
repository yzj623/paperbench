#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 PAPER_ID LEGACY_OUTLINE_JSON [OUTPUT_DIR [--resume]]" >&2
  exit 2
fi
RESUME=false
if [[ $# -eq 4 ]]; then
  [[ "$4" == "--resume" ]] || { echo "Fourth argument must be --resume" >&2; exit 2; }
  RESUME=true
fi

PAPER_ID="$1"
OUTLINE_FILE="$2"
if [[ ! "$PAPER_ID" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "Invalid PaperBench paper ID: $PAPER_ID" >&2
  exit 2
fi
[[ -f "$OUTLINE_FILE" ]] || {
  echo "Legacy outline does not exist: $OUTLINE_FILE" >&2
  exit 2
}

RESEARCH_AGENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPERBENCH_ROOT="${PAPERBENCH_ROOT:-$(cd "$RESEARCH_AGENT_ROOT/../.." && pwd)/PaperBench-Official}"
PAPER_DIR="$PAPERBENCH_ROOT/project/paperbench/data/papers/$PAPER_ID"
INSTRUCTIONS="$PAPERBENCH_ROOT/project/paperbench/paperbench/instructions/code_only_instructions.txt"
PYTHON="$RESEARCH_AGENT_ROOT/.venv-wsl/bin/python"

[[ -x "$PYTHON" ]] || {
  echo "Missing local Python environment. Run: $RESEARCH_AGENT_ROOT/scripts/setup.sh" >&2
  exit 1
}
if [[ -f "$RESEARCH_AGENT_ROOT/.env" ]]; then
  set -a
  source "$RESEARCH_AGENT_ROOT/.env"
  set +a
fi
agent_key_env=DASHSCOPE_API_KEY
if [[ "${RESEARCH_AGENT_API_CONFIG:-${AGENT_API_CONFIG:-qwen37flash}}" == deepseekv4flash ]]; then
  agent_key_env=DEEPSEEK_API_KEY
fi
[[ -n "${!agent_key_env:-}" ]] || {
  echo "Set $agent_key_env or provide it in $RESEARCH_AGENT_ROOT/.env" >&2
  exit 1
}
[[ -d "$PAPER_DIR" ]] || {
  echo "Unknown PaperBench paper ID: $PAPER_ID" >&2
  exit 1
}
[[ -f "$INSTRUCTIONS" ]] || {
  echo "Missing public Code-Dev instructions: $INSTRUCTIONS" >&2
  exit 1
}

for required in paper.md addendum.md; do
  [[ -f "$PAPER_DIR/$required" ]] || {
    echo "Missing public paper input: $PAPER_DIR/$required" >&2
    exit 1
  }
  if grep -F -m 1 'version https://git-lfs.github.com/spec/v1' \
    "$PAPER_DIR/$required" >/dev/null; then
    echo "Paper input is still a Git LFS pointer: $PAPER_DIR/$required" >&2
    echo "Hydrate this paper from the official PaperBench checkout first." >&2
    exit 1
  fi
done

if [[ "$RESUME" == true ]]; then
  RUN_ROOT="$3"
  [[ -f "$RUN_ROOT/research_agent/checklist_enrichment_state.json" ]] || {
    echo "No resumable checklist state in: $RUN_ROOT" >&2
    exit 1
  }
elif [[ $# -ge 3 ]]; then
  RUN_ROOT="$3"
  mkdir -p "$(dirname "$RUN_ROOT")"
  mkdir "$RUN_ROOT"  # Refuse to overwrite a previous planner test.
else
  mkdir -p "$RESEARCH_AGENT_ROOT/plan-tests/$PAPER_ID"
  RUN_ROOT="$(mktemp -d "$RESEARCH_AGENT_ROOT/plan-tests/$PAPER_ID/run.XXXXXXXX")"
fi

WORKSPACE="$RUN_ROOT/workspace"
if [[ "$RESUME" == false ]]; then
  mkdir -p "$WORKSPACE/paper/assets" "$WORKSPACE/submission"
  cp "$PAPER_DIR/paper.md" "$WORKSPACE/paper/paper.md"
  cp "$PAPER_DIR/addendum.md" "$WORKSPACE/paper/addendum.md"
  cp "$INSTRUCTIONS" "$WORKSPACE/instructions.txt"
  if [[ -d "$PAPER_DIR/assets" ]]; then
    cp -a "$PAPER_DIR/assets/." "$WORKSPACE/paper/assets/"
  fi
fi

echo "Planner-only test for $PAPER_ID"
echo "Output directory: $RUN_ROOT"
PYTHONPATH="$RESEARCH_AGENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m research_agent.plan_enrichment_runner \
  --workspace "$WORKSPACE" \
  --outline "$OUTLINE_FILE" \
  --run-dir "$RUN_ROOT/research_agent" \
  --model "${RESEARCH_AGENT_MODEL:-qwen3.7-flash}"
