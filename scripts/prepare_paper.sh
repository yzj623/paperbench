#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 PAPER_ID" >&2
  exit 2
fi

PAPER_ID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAPERBENCH_ROOT="${PAPERBENCH_ROOT:-$REPO_ROOT/PaperBench-Official}"
PAPER_DIR="$PAPERBENCH_ROOT/project/paperbench/data/papers/$PAPER_ID"

[[ "$PAPER_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid paper id" >&2; exit 2; }
[[ -f "$PAPER_DIR/config.yaml" ]] || { echo "Unknown paper id: $PAPER_ID" >&2; exit 2; }

needs_fetch=0
for name in paper.pdf paper.md addendum.md blacklist.txt; do
  if [[ ! -f "$PAPER_DIR/$name" ]] || \
    grep -F -m 1 'version https://git-lfs.github.com/spec/v1' "$PAPER_DIR/$name" >/dev/null 2>&1; then
    needs_fetch=1
  fi
done

if [[ "$needs_fetch" -eq 1 ]]; then
  command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
  git lfs version >/dev/null 2>&1 || { echo "git-lfs is required" >&2; exit 1; }
  UPSTREAM_CACHE="${PAPERBENCH_UPSTREAM_CACHE:-$REPO_ROOT/.runtime/frontier-evals}"
  if [[ ! -d "$UPSTREAM_CACHE/.git" ]]; then
    mkdir -p "$(dirname "$UPSTREAM_CACHE")"
    GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1 timeout "${PAPERBENCH_CLONE_TIMEOUT:-300}" \
      git clone --filter=blob:none --sparse \
      https://github.com/openai/frontier-evals.git "$UPSTREAM_CACHE"
  fi
  git -C "$UPSTREAM_CACHE" sparse-checkout add "project/paperbench/data/papers/$PAPER_ID"
  timeout "${PAPERBENCH_LFS_TIMEOUT:-300}" git -C "$UPSTREAM_CACHE" lfs pull \
    --include="project/paperbench/data/papers/$PAPER_ID/**" --exclude=""
  upstream_dir="$UPSTREAM_CACHE/project/paperbench/data/papers/$PAPER_ID"
  mkdir -p "$PAPER_DIR"
  for name in paper.pdf paper.md addendum.md blacklist.txt; do
    cp -f -- "$upstream_dir/$name" "$PAPER_DIR/$name"
  done
  if [[ -d "$upstream_dir/assets" ]]; then
    mkdir -p "$PAPER_DIR/assets"
    cp -a -- "$upstream_dir/assets/." "$PAPER_DIR/assets/"
  fi
fi

for name in paper.pdf paper.md addendum.md blacklist.txt; do
  [[ -f "$PAPER_DIR/$name" ]] || { echo "Missing input: $PAPER_DIR/$name" >&2; exit 1; }
  if grep -F -m 1 'version https://git-lfs.github.com/spec/v1' "$PAPER_DIR/$name" >/dev/null; then
    echo "Git LFS pointer remains: $PAPER_DIR/$name" >&2
    exit 1
  fi
done
echo "Paper inputs ready: $PAPER_DIR"
