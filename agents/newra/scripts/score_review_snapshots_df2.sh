#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 2
fi

run_dir="$(realpath "$1")"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
snapshot_dir="$run_dir/agent-run/research_agent/full_review/snapshots"
output_root="$run_dir/evaluation/review-snapshots-df2-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$output_root"
printf '%s\n' "$output_root" > "$run_dir/evaluation/LATEST_REVIEW_SNAPSHOTS_DF2"

for index in 000 001 002; do
  archive="$snapshot_dir/submission-$index.tar.gz"
  [[ -f "$archive" ]] || { echo "Missing $archive" >&2; exit 1; }
  stage="$output_root/stage-$index"
  result="$output_root/score-$index"
  mkdir -p "$stage"
  tar -xzf "$archive" -C "$stage"
  echo "$(date -Is) START snapshot=$index result=$result" | tee -a "$output_root/status.log"
  if PAPERBENCH_GRADER_API_CONFIG=ustc-deepseek-flash \
      "$script_dir/evaluate_submission.sh" all-in-one "$stage/submission" "$result" \
      > "$output_root/score-$index.log" 2>&1; then
    echo "$(date -Is) DONE snapshot=$index result=$result" | tee -a "$output_root/status.log"
  else
    code=$?
    echo "$(date -Is) FAILED snapshot=$index exit=$code result=$result" | tee -a "$output_root/status.log"
  fi
done

echo "$(date -Is) FINISHED" | tee -a "$output_root/status.log"
