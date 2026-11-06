#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${CODEX_IMAGE:-codex-paperbench:latest}"
BUILD_JOBS="${BUILD_JOBS:-1}"
BUILD_PROXY="${CODEX_BUILD_PROXY:-}"
BUILD_CGROUP_PARENT="${CODEX_BUILD_CGROUP_PARENT:-}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --jobs) BUILD_JOBS="$2"; shift 2 ;;
    --proxy) BUILD_PROXY="$2"; shift 2 ;;
    --cgroup-parent) BUILD_CGROUP_PARENT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Usage: $0 [--image TAG] [--jobs N] [--proxy URL] [--cgroup-parent GROUP] [--dry-run]" >&2; exit 2 ;;
  esac
done
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'jobs must be a positive integer' >&2; exit 2; }
command=(docker build --file "$PROJECT_ROOT/Dockerfile.codex" --build-arg "BUILD_JOBS=$BUILD_JOBS" --tag "$IMAGE" "$PROJECT_ROOT")
if [[ -n "$BUILD_CGROUP_PARENT" ]]; then
  command+=(--cgroup-parent "$BUILD_CGROUP_PARENT")
fi
if [[ -n "$BUILD_PROXY" ]]; then
  # Docker's predefined proxy arguments apply to build RUN steps, not the daemon.
  # Do not persist proxy addresses in the final image's ENV.
  command+=(--build-arg "HTTP_PROXY=$BUILD_PROXY" --build-arg "HTTPS_PROXY=$BUILD_PROXY"
            --build-arg "http_proxy=$BUILD_PROXY" --build-arg "https_proxy=$BUILD_PROXY"
            --build-arg "NO_PROXY=localhost,127.0.0.1" --build-arg "no_proxy=localhost,127.0.0.1")
fi
if [[ "$DRY_RUN" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; else exec "${command[@]}"; fi
