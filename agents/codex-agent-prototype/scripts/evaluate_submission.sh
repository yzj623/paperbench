#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 PAPER_ID SUBMISSION_DIR OUTPUT_DIR" >&2
  exit 2
fi

PAPER_ID="$1"
SUBMISSION_DIR="$(realpath "$2")"
OUTPUT_DIR="$(realpath -m "$3")"
CODEX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPERBENCH_ROOT="${PAPERBENCH_ROOT:-$(realpath "$CODEX_ROOT/vendor/paperbench-official")}"
REQUESTED_GRADER_API_CONFIG="${PAPERBENCH_GRADER_API_CONFIG:-}"
if [[ -f "$CODEX_ROOT/.grader.env" ]]; then
  set -a
  source "$CODEX_ROOT/.grader.env"
  set +a
fi

GRADER_API_CONFIG="${REQUESTED_GRADER_API_CONFIG:-${PAPERBENCH_GRADER_API_CONFIG:-ustc-deepseek-flash}}"
grader_config_args=()
case "$GRADER_API_CONFIG" in
  openai-oauth|gpt6-luna|gpt-6-luna)
    GRADER_API_CONFIG=openai-oauth
    credentials_dir="${CODEX_GRADER_OPENAI_OAUTH_DIR:-${OPENAI_OAUTH_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/ppblab/openai-oauth}}"
    [[ -f "$credentials_dir/credentials.json" ]] || {
      echo "Shared OAuth credentials missing: $credentials_dir/credentials.json" >&2
      exit 1
    }
    export GRADER_OPENAI_API_KEY="configured-via-shared-oauth"
    grader_config_args+=(
      paperbench.judge.completer_config=codex_eval.openai_judge:OpenAIOAuthJudgeCompleterConfig
    )
    if [[ -n "${CODEX_GRADER_REASONING_EFFORT:-}" ]]; then
      grader_config_args+=("paperbench.judge.completer_config.reasoning_effort=$CODEX_GRADER_REASONING_EFFORT")
    fi
    ;;
  openai)
    [[ -n "${GRADER_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}" ]] || {
      echo "GRADER_OPENAI_API_KEY or OPENAI_API_KEY is required by the OpenAI judge." >&2
      exit 1
    }
    ;;
  ustc-deepseek-flash)
    [[ -n "${USTC_LLM_API_KEY:-}" ]] || {
      echo "USTC_LLM_API_KEY is required by the USTC DeepSeek-Flash judge." >&2
      exit 1
    }
    export GRADER_OPENAI_API_KEY="configured-via-ustc"
    grader_config_args+=(
      paperbench.judge.completer_config=codex_eval.ustc_judge:USTCDeepSeekFlashJudgeCompleterConfig
    )
    if [[ -n "${USTC_JUDGE_MODEL:-}" ]]; then
      grader_config_args+=("paperbench.judge.completer_config.model=$USTC_JUDGE_MODEL")
    fi
    ;;
  deepseek-official-flash)
    [[ -n "${DEEPSEEK_API_KEY:-}" ]] || {
      echo "DEEPSEEK_API_KEY is required by the official DeepSeek Flash judge." >&2
      exit 1
    }
    export GRADER_OPENAI_API_KEY="configured-via-deepseek"
    grader_config_args+=(
      paperbench.judge.completer_config=codex_eval.deepseek_official_judge:DeepSeekOfficialFlashJudgeCompleterConfig
    )
    ;;
  *)
    echo "Unknown PAPERBENCH_GRADER_API_CONFIG: $GRADER_API_CONFIG" >&2
    exit 2
    ;;
esac
export PAPERBENCH_GRADER_API_CONFIG="$GRADER_API_CONFIG"

[[ -d "$SUBMISSION_DIR" ]] || {
  echo "Submission directory does not exist: $SUBMISSION_DIR" >&2
  exit 1
}
[[ -f "$PAPERBENCH_ROOT/project/paperbench/pyproject.toml" ]] || {
  echo "Official PaperBench project not found: $PAPERBENCH_ROOT" >&2
  exit 1
}
SPLIT_FILE="$PAPERBENCH_ROOT/project/paperbench/experiments/splits/all.txt"
[[ -f "$SPLIT_FILE" ]] || {
  echo "Official PaperBench split file is missing: $SPLIT_FILE" >&2
  exit 1
}
awk -v paper_id="$PAPER_ID" '{ sub(/\r$/, ""); if ($0 == paper_id) found = 1 } END { exit !found }' "$SPLIT_FILE" || {
  echo "Paper ID is not in the official all split: $PAPER_ID" >&2
  exit 1
}
[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "Evaluation output already exists: $OUTPUT_DIR" >&2
  exit 1
}

STAGING="$OUTPUT_DIR/direct-submissions"
RUNS_DIR="$OUTPUT_DIR/runs"
mkdir -p "$STAGING/$PAPER_ID" "$RUNS_DIR"
# Point PaperBench at the original directory instead of copying its contents.
ln -s "$SUBMISSION_DIR" "$STAGING/$PAPER_ID/submission"

cd "$PAPERBENCH_ROOT/project/paperbench"
PYTHONPATH="$CODEX_ROOT" \
UV_PROJECT_ENVIRONMENT="$CODEX_ROOT/.venv-paperbench" \
PAPERBENCH_JUDGE_LOG_DIR="$OUTPUT_DIR/judge-logs" \
uv run --frozen python -c \
  'import chz; from nanoeval.setup import nanoeval_entrypoint; from codex_eval.eval_entrypoint import main; nanoeval_entrypoint(chz.entrypoint(main))' \
  paperbench.paper_id="$PAPER_ID" \
  paperbench.paper_split=all \
  paperbench.n_tries=1 \
  paperbench.runs_dir="$RUNS_DIR" \
  paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
  paperbench.solver.submissions_dir="$STAGING" \
  paperbench.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
  paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
  paperbench.solver.computer_runtime.env.pull_from_registry=false \
  paperbench.judge.code_only=true \
  "${grader_config_args[@]}" \
  paperbench.reproduction.skip_reproduction=true \
  runner.concurrency=1 \
  runner.max_retries=3 \
  runner.enable_slackbot=false \
  runner.should_backup=false \
  runner.recorder=nanoeval.json_recorder:json_recorder

echo "Official PaperBench Code-Dev evaluation complete."
echo "Evaluation output: $OUTPUT_DIR"
