#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 PAPER_ID SUBMISSION_DIR [SERIES_PARENT]" >&2
  exit 2
fi

PAPER_ID="$1"
SUBMISSION_DIR="$(realpath -e "$2")"
SERIES_PARENT="${3:-$(dirname "$SUBMISSION_DIR")/serial-regrades}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ "$PAPER_ID" =~ ^[a-z0-9][a-z0-9-]*$ ]] || {
  echo "Invalid PaperBench paper ID: $PAPER_ID" >&2
  exit 2
}
[[ -d "$SUBMISSION_DIR" ]] || {
  echo "Submission is not a directory: $SUBMISSION_DIR" >&2
  exit 2
}
mkdir -p "$SERIES_PARENT"
SERIES_PARENT="$(realpath -e "$SERIES_PARENT")"
STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
SERIES_ROOT="$(mktemp -d "$SERIES_PARENT/$STAMP-serial-five-regrades.XXXXXX")"
printf '%s\n' "$SERIES_ROOT" > "$SERIES_PARENT/LAST_SERIAL_REGRADE_PATH"
printf 'series_root=%s\nsubmission=%s\npaper_id=%s\n' \
  "$SERIES_ROOT" "$SUBMISSION_DIR" "$PAPER_ID"

failed=0
for round in 1 2 3 4 5; do
  ROUND_ROOT="$(mktemp -d "$SERIES_ROOT/round-${round}.XXXXXX")"
  printf 'round=%s status=started output=%s\n' "$round" "$ROUND_ROOT" >> "$SERIES_ROOT/manifest.txt"
  printf 'Starting round %s/5: %s\n' "$round" "$ROUND_ROOT"

  if bash "$SCRIPT_DIR/evaluate_submission.sh" \
    "$PAPER_ID" "$SUBMISSION_DIR" "$ROUND_ROOT" > "$ROUND_ROOT/evaluator.log" 2>&1; then
    printf 'round=%s status=complete output=%s\n' "$round" "$ROUND_ROOT" >> "$SERIES_ROOT/manifest.txt"
    printf 'Completed round %s/5\n' "$round"
  else
    status=$?
    printf 'round=%s status=failed exit=%s output=%s\n' \
      "$round" "$status" "$ROUND_ROOT" >> "$SERIES_ROOT/manifest.txt"
    printf 'Round %s/5 failed (exit %s); see %s/evaluator.log\n' \
      "$round" "$status" "$ROUND_ROOT" >&2
    failed=1
  fi
done

printf 'All five rounds attempted. Results: %s\n' "$SERIES_ROOT"
exit "$failed"
