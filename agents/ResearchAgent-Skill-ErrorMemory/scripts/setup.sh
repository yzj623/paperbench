#!/usr/bin/env bash
set -euo pipefail

RESEARCH_AGENT_ROOT="${RESEARCH_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PAPERBENCH_ROOT="${PAPERBENCH_ROOT:-$(cd "$RESEARCH_AGENT_ROOT/../.." && pwd)/PaperBench-Official}"
PAPERBENCH_PROJECT="$PAPERBENCH_ROOT/project/paperbench"

command -v uv >/dev/null || {
  echo "uv is required inside WSL." >&2
  exit 1
}
command -v docker >/dev/null || {
  echo "Docker is required inside WSL." >&2
  exit 1
}
git lfs version >/dev/null 2>&1 || {
  echo "git-lfs is required inside WSL." >&2
  exit 1
}

test -f "$PAPERBENCH_PROJECT/uv.lock" || {
  echo "Official PaperBench checkout not found at: $PAPERBENCH_ROOT" >&2
  exit 1
}

UV_PROJECT_ENVIRONMENT="$RESEARCH_AGENT_ROOT/.venv-wsl" \
  uv sync --project "$PAPERBENCH_PROJECT" --frozen

if ! docker image inspect pb-env:latest >/dev/null 2>&1; then
  docker build --platform=linux/amd64 -t pb-env:latest \
    -f "$PAPERBENCH_PROJECT/paperbench/Dockerfile.base" \
    "$PAPERBENCH_PROJECT"
fi

echo "ResearchAgent controller environment and official pb-env image are ready."
