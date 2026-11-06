# ResearchAgent with persistent checker error records

This project keeps the ResearchAgent implementation outside the official
PaperBench checkout. The only boundary is the benchmark's public task input
and canonical submission directory:

1. `scripts/run_agent.sh PAPER_ID` hydrates only that paper's official Git LFS
   input, creates an isolated `pb-env` Docker workspace, and runs ResearchAgent.
2. The rollout exports a standard `submission/` directory.
3. `scripts/evaluate_submission.sh PAPER_ID SUBMISSION_DIR` passes that folder
   to PaperBench's unmodified `PBDirectSubmissionSolver` and Code-Dev judge.
4. `scripts/run_experiment.sh PAPER_ID [OUTLINE_JSON]` runs both stages in order,
   optionally skipping planning and starting from a supplied frozen outline.

The official checkout is expected at `./PaperBench-Official`, this project
at `./agents/ResearchAgent-Skill-ErrorMemory`, and outputs at `./experiments`.
Override them with `PAPERBENCH_ROOT`, `RESEARCH_AGENT_ROOT`, and
`EXPERIMENTS_ROOT`.

Run from WSL/Linux with Docker available:

```bash
cd agents/ResearchAgent-Skill-ErrorMemory
bash scripts/setup.sh
bash scripts/run_experiment.sh mechanistic-understanding
```

To run with a fixed outline, pass the JSON plan as the second argument. The
format is the same as `plans/all-in-one.plan.json` (`subtasks`, each containing
`id`, `name`, and `description`):

```bash
bash scripts/run_experiment.sh all-in-one plans/all-in-one.plan.json
```

The script validates the outline, copies it into that run's directory as
`input_outline.json` for provenance, and passes the copy to the agent. The
planner phase is skipped; the supplied subtask list is used as the frozen plan.

Skills are enabled by default. To run the same ErrorMemory controller without
skill selection or Skill messages, set `RESEARCH_AGENT_ENABLE_SKILLS=0`:

```bash
RESEARCH_AGENT_ENABLE_SKILLS=0 bash scripts/run_experiment.sh all-in-one plans/all-in-one.plan.json
```

Use `RESEARCH_AGENT_ENABLE_SKILLS=1` to enable them explicitly. The chosen mode
is recorded in `agent-run/skill-config.json` for each run.

The project `.env` supplies the configured agent and grader API credentials.
The first `run_experiment.sh PAPER_ID` invocation prepares the local controller
environment and PaperBench base image when either is missing, then runs the
agent and Code-Dev evaluation. It creates a unique run directory under
`EXPERIMENTS_ROOT/PAPER_ID/`.

The copied `.env` currently selects the Qwen3.7-Flash agent and USTC
DeepSeek-Flash judge configuration. The judge uses PaperBench's existing
Code-Dev scoring path. Optional provider adapters and checker changes live in
this project; no ResearchAgent source directory is imported at runtime.

To use the optional official Zhipu GLM-4.7-Flash agent configuration while
leaving the PaperBench judge unchanged:

```bash
export RESEARCH_AGENT_API_CONFIG=glm47flash
export ZAI_API_KEY=...
export GRADER_OPENAI_API_KEY=...
bash scripts/run_experiment.sh pinn
```

This selects model `glm-4.7-flash` at the official Zhipu endpoint. Override the
model only when deliberately testing another compatible deployment with
`RESEARCH_AGENT_MODEL`.

To use USTC DeepSeek-Flash for the Planner, Main Agent, and Check Agent:

```bash
export RESEARCH_AGENT_API_CONFIG=ustc-deepseek-flash
export USTC_LLM_API_KEY=...
export GRADER_OPENAI_API_KEY=...
bash scripts/run_experiment.sh pinn
```

This selects the currently available model ID `deepseek-flash` at
`https://api.llm.ustc.edu.cn/v1` with its documented 1M context window.

To use Alibaba Cloud Qwen3.7-Flash with thinking disabled:

```bash
export RESEARCH_AGENT_API_CONFIG=qwen37flash
export DASHSCOPE_API_KEY=...
export GRADER_OPENAI_API_KEY=...
bash scripts/run_experiment.sh pinn
```

This selects `qwen3.7-flash` through the Beijing OpenAI-compatible DashScope
endpoint. The adapter explicitly sends `enable_thinking=false`; the PaperBench
judge remains unchanged unless separately configured.

To use the OpenAI ChatGPT-plan OAuth path with `gpt-6-luna`:

```bash
.venv-wsl/bin/python scripts/openai_oauth.py login --probe-model gpt-6-luna
RESEARCH_AGENT_API_CONFIG=openai-oauth \
RESEARCH_AGENT_MODEL=gpt-6-luna \
bash scripts/run_experiment.sh all-in-one
```

This path uses the same OAuth credential store, Responses adapter, and retry
policy as the other ResearchAgent variants. See
`scripts/OPENAI_OAUTH.md` and `scripts/OPENAI_RETRY_POLICY.md`.

The classic OpenAI API-key path is also available:

```bash
RESEARCH_AGENT_API_CONFIG=openai \
RESEARCH_AGENT_MODEL=gpt-4.1-mini \
OPENAI_API_KEY=... \
bash scripts/run_experiment.sh all-in-one
```

The evaluator remains on PaperBench's OpenAI judge by default. To route all
judge calls (including integer/float structured helper calls) through USTC's
DeepSeek-Flash without changing PaperBench's prompts or scoring logic:

```bash
export PAPERBENCH_GRADER_API_CONFIG=ustc-deepseek-flash
export USTC_LLM_API_KEY=...
bash scripts/evaluate_submission.sh pinn /path/to/submission
```
