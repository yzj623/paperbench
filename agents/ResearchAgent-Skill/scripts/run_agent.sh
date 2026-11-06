#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 PAPER_ID [OUTPUT_DIR]" >&2
  exit 2
fi

PAPER_ID="$1"
RESEARCH_AGENT_ROOT="${RESEARCH_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "$RESEARCH_AGENT_ROOT/.env" ]]; then
  set -a
  source "$RESEARCH_AGENT_ROOT/.env"
  set +a
fi
PAPERBENCH_ROOT="${PAPERBENCH_ROOT:-$(cd "$RESEARCH_AGENT_ROOT/../.." && pwd)/PaperBench-Official}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$(cd "$RESEARCH_AGENT_ROOT/../.." && pwd)/experiments}"
API_CONFIG="${RESEARCH_AGENT_API_CONFIG:-${AGENT_API_CONFIG:-qwen37flash}}"
case "$API_CONFIG" in
  openai-oauth)
    MODEL="${RESEARCH_AGENT_MODEL:-gpt-6-luna}"
    API_KEY_ENV=""
    export RESEARCH_AGENT_OPENAI_OAUTH_DIR="${RESEARCH_AGENT_OPENAI_OAUTH_DIR:-${OPENAI_OAUTH_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/ppblab/openai-oauth}}"
    [[ -f "$RESEARCH_AGENT_OPENAI_OAUTH_DIR/credentials.json" ]] || {
      echo "OAuth credentials missing; run scripts/openai_oauth.py login first." >&2
      exit 1
    }
    ;;
  openai)
    MODEL="${RESEARCH_AGENT_MODEL:-gpt-4.1-mini}"
    API_KEY_ENV=OPENAI_API_KEY
    ;;
  glm47flash)
    MODEL="${RESEARCH_AGENT_MODEL:-glm-4.7-flash}"
    API_KEY_ENV=ZAI_API_KEY
    ;;
  ustc-deepseek-flash)
    MODEL="${RESEARCH_AGENT_MODEL:-deepseek-flash}"
    API_KEY_ENV=USTC_LLM_API_KEY
    ;;
  deepseekv4flash)
    MODEL="${RESEARCH_AGENT_MODEL:-deepseek-v4-flash}"
    API_KEY_ENV=DEEPSEEK_API_KEY
    ;;
  qwen37flash)
    MODEL="${RESEARCH_AGENT_MODEL:-qwen3.7-flash}"
    API_KEY_ENV=DASHSCOPE_API_KEY
    ;;
  *)
    echo "Unknown RESEARCH_AGENT_API_CONFIG: $API_CONFIG" >&2
    exit 2
    ;;
esac
STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
RUN_ROOT="${2:-$EXPERIMENTS_ROOT/$PAPER_ID/$STAMP}"
WORKSPACE="$RUN_ROOT/workspace"
RUN_DIR="$RUN_ROOT/agent-run"
RUN_ID="${PAPER_ID}_$(python3 -c 'import uuid; print(uuid.uuid4())')"
RUN_GROUP_ID="${STAMP}_run-group_researchagent-skill"
CONTAINER="pb-ra-${PAPER_ID//[^a-zA-Z0-9_.-]/-}-${RANDOM}-${RANDOM}"
PAPER_DIR="$PAPERBENCH_ROOT/project/paperbench/data/papers/$PAPER_ID"
INSTRUCTIONS="$PAPERBENCH_ROOT/project/paperbench/paperbench/instructions/code_only_instructions.txt"

if [[ -n "$API_KEY_ENV" ]]; then
[[ -n "${!API_KEY_ENV:-}" ]] || {
  echo "$API_KEY_ENV is not set for ResearchAgent API config '$API_CONFIG'." >&2
  exit 1
}
fi
"$(cd "$RESEARCH_AGENT_ROOT/../.." && pwd)/scripts/prepare_paper.sh" "$PAPER_ID"

[[ -f "$PAPER_DIR/config.yaml" ]] || {
  echo "Unknown PaperBench paper id: $PAPER_ID" >&2
  exit 1
}
for required in paper.pdf paper.md addendum.md blacklist.txt; do
  [[ -f "$PAPER_DIR/$required" ]] || {
    echo "Missing official input: $PAPER_DIR/$required" >&2
    exit 1
  }
  # Read the file directly: with pipefail, `head | grep -q` can report SIGPIPE
  # after a match and incorrectly treat an LFS pointer as real content.
  if grep -F -m 1 'version https://git-lfs.github.com/spec/v1' \
    "$PAPER_DIR/$required" >/dev/null; then
    echo "Official input is still a Git LFS pointer: $PAPER_DIR/$required" >&2
    exit 1
  fi
done

mkdir -p "$WORKSPACE/paper/assets" "$WORKSPACE/submission" "$WORKSPACE/logs" "$WORKSPACE/agent" "$RUN_DIR"
cp "$PAPER_DIR/paper.pdf" "$WORKSPACE/paper/paper.pdf"
cp "$PAPER_DIR/paper.md" "$WORKSPACE/paper/paper.md"
cp "$PAPER_DIR/addendum.md" "$WORKSPACE/paper/addendum.md"
cp "$PAPER_DIR/blacklist.txt" "$WORKSPACE/paper/blacklist.txt"
cp "$INSTRUCTIONS" "$WORKSPACE/instructions.txt"
if [[ -d "$PAPER_DIR/assets" ]]; then
  cp -a "$PAPER_DIR/assets/." "$WORKSPACE/paper/assets/"
fi
AGENT_ENV="$PAPERBENCH_ROOT/project/paperbench/paperbench/solvers/agent.env"
if [[ -f "$AGENT_ENV" ]]; then
  cp "$AGENT_ENV" "$WORKSPACE/agent.env"
fi

mkdir -p "$WORKSPACE/.vscode" "$WORKSPACE/submission/.git/hooks"
cp "$PAPERBENCH_ROOT/project/paperbench/paperbench/solvers/launch.json" \
  "$WORKSPACE/.vscode/launch.json"
cp "$PAPERBENCH_ROOT/project/paperbench/paperbench/solvers/pre-commit" \
  "$WORKSPACE/submission/.git/hooks/pre-commit"
chmod +x "$WORKSPACE/submission/.git/hooks/pre-commit"

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker run -d --rm --name "$CONTAINER" \
  --workdir /home \
  --network bridge \
  -e OPENAI_API_KEY \
  -e SUBMISSION_DIR=/home/submission \
  -e LOGS_DIR=/home/logs \
  -e AGENT_DIR=/home/agent \
  -v "$WORKSPACE:/home" \
  -v "$WORKSPACE:$WORKSPACE" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  pb-env:latest sleep infinity >/dev/null

docker exec "$CONTAINER" bash -lc \
  "git config --global --add safe.directory /home/submission && \
   cd /home/submission && git init && \
   git config user.email agent@example.com && git config user.name agent"

PYTHONPATH="$RESEARCH_AGENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$RESEARCH_AGENT_ROOT/.venv-wsl/bin/python" -m research_agent.runner \
  --container "$CONTAINER" \
  --workspace "$WORKSPACE" \
  --run-dir "$RUN_DIR" \
  --run-id "$RUN_ID" \
  --run-group-id "$RUN_GROUP_ID" \
  --skill-target "$PAPER_ID" \
  --api-config "$API_CONFIG" \
  --model "$MODEL" \
  ${RESEARCH_AGENT_PLAN:+--plan "$RESEARCH_AGENT_PLAN"}

LATEST_ARCHIVE="$(find "$RUN_DIR/submissions" -type f -name submission.tar.gz -print | sort | tail -n 1)"
[[ -n "$LATEST_ARCHIVE" ]] || {
  echo "ResearchAgent finished without a canonical submission snapshot." >&2
  exit 1
}
mkdir -p "$RUN_ROOT/export"
tar -xzf "$LATEST_ARCHIVE" -C "$RUN_ROOT/export"
[[ -d "$RUN_ROOT/export/submission" ]] || {
  echo "Canonical snapshot did not contain submission/." >&2
  exit 1
}
printf '%s\n' "$RUN_ROOT/export/submission" > "$RUN_ROOT/FINAL_SUBMISSION_PATH"
printf '%s\n' "$RUN_ROOT" > "$EXPERIMENTS_ROOT/LAST_AGENT_RUN"

echo "ResearchAgent rollout complete."
echo "Run root: $RUN_ROOT"
echo "Submission: $RUN_ROOT/export/submission"
