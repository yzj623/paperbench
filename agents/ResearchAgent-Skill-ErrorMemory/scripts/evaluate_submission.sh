#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 PAPER_ID SUBMISSION_DIR [OUTPUT_DIR]" >&2
  exit 2
fi

PAPER_ID="$1"
SUBMISSION_DIR="$(realpath "$2")"
RESEARCH_AGENT_ROOT="${RESEARCH_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GRADER_API_OVERRIDE="${PAPERBENCH_GRADER_API_CONFIG:-}"
if [[ -f "$RESEARCH_AGENT_ROOT/.env" ]]; then
  set -a
  source "$RESEARCH_AGENT_ROOT/.env"
  set +a
fi
if [[ -n "$GRADER_API_OVERRIDE" ]]; then
  export PAPERBENCH_GRADER_API_CONFIG="$GRADER_API_OVERRIDE"
fi
PAPERBENCH_ROOT="${PAPERBENCH_ROOT:-$(cd "$RESEARCH_AGENT_ROOT/../.." && pwd)/PaperBench-Official}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$(cd "$RESEARCH_AGENT_ROOT/../.." && pwd)/experiments}"
STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUTPUT_DIR="${3:-$EXPERIMENTS_ROOT/$PAPER_ID/evaluations/$STAMP}"
STAGING="$OUTPUT_DIR/direct-submissions"
RUNS_DIR="$OUTPUT_DIR/runs"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

GRADER_API_CONFIG="${PAPERBENCH_GRADER_API_CONFIG:-ustc-deepseek-flash}"
grader_config_args=()
case "$GRADER_API_CONFIG" in
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
    # PaperBench's stock preflight checks this variable before constructing
    # the selected completer. Keep it process-local; the USTC adapter reads
    # the actual credential only from USTC_LLM_API_KEY.
    export GRADER_OPENAI_API_KEY="configured-via-ustc"
    grader_config_args+=(
      paperbench.judge.completer_config=research_agent.ustc_judge:USTCDeepSeekFlashJudgeCompleterConfig
    )
    ;;
  deepseek-official-flash)
    [[ -n "${DEEPSEEK_API_KEY:-}" ]] || {
      echo "DEEPSEEK_API_KEY is required by the official DeepSeek Flash judge." >&2
      exit 1
    }
    export GRADER_OPENAI_API_KEY="configured-via-deepseek"
    grader_config_args+=(
      paperbench.judge.completer_config=research_agent.deepseek_official_judge:DeepSeekOfficialFlashJudgeCompleterConfig
    )
    ;;
  *)
    echo "Unknown PAPERBENCH_GRADER_API_CONFIG: $GRADER_API_CONFIG" >&2
    exit 2
    ;;
esac
[[ -d "$SUBMISSION_DIR" ]] || {
  echo "Submission directory does not exist: $SUBMISSION_DIR" >&2
  exit 1
}

mkdir -p "$STAGING/$PAPER_ID/submission" "$RUNS_DIR"
cp -a "$SUBMISSION_DIR/." "$STAGING/$PAPER_ID/submission/"

cd "$PAPERBENCH_ROOT/project/paperbench"
PYTHONPATH="$RESEARCH_AGENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
UV_PROJECT_ENVIRONMENT="$RESEARCH_AGENT_ROOT/.venv-wsl" \
"$UV_BIN" run --frozen python -c \
  'import chz; from nanoeval.setup import nanoeval_entrypoint; from research_agent.eval_entrypoint import main; nanoeval_entrypoint(chz.entrypoint(main))' \
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
