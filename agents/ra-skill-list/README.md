# Standalone ResearchAgent for PaperBench

This project keeps the ResearchAgent implementation outside the official
PaperBench checkout. The only boundary is the benchmark's public task input
and canonical submission directory:

1. `scripts/run_agent.sh PAPER_ID` hydrates only that paper's official Git LFS
   input, creates an isolated `pb-env` Docker workspace, and runs ResearchAgent.
2. The rollout exports a standard `submission/` directory.
3. `scripts/evaluate_submission.sh PAPER_ID SUBMISSION_DIR` passes that folder
   to PaperBench's unmodified `PBDirectSubmissionSolver` and Code-Dev judge.
4. `scripts/run_experiment.sh PAPER_ID PLAN_JSON` accepts either a no-checklist
   outline (enriched before execution) or a complete checklist plan (used as-is),
   runs the agent, and grades its frozen submission.

The official checkout is expected at `./PaperBench-Official`, this project
at `./agents/ra-skill-list`, and outputs at `./experiments`.
Override them with `PAPERBENCH_ROOT`, `RESEARCH_AGENT_ROOT`, and
`EXPERIMENTS_ROOT`.

Skill selection is enabled by default. To run the same checklist workflow without
Skill selection or injected Skill text, set `RESEARCH_AGENT_ENABLE_SKILLS=0`:

```bash
RESEARCH_AGENT_ENABLE_SKILLS=0 bash scripts/run_experiment.sh all-in-one plans/all-in-one.plan.json
```

The same switch works with `scripts/run_agent.sh` and `scripts/resume_experiment.sh`.
The default Checker materials mode is `tools`, matching ordinary ResearchAgent
paper access; `embedded` remains an optional diagnostic mode.

Checker paper access has two selectable modes. `tools` is the default and uses
the original read-only paper-file tools. `embedded` places complete `paper.md`,
`addendum.md`, and checker-only `check_instruction.txt` text before the current
subtask in protected messages; the checker's file tools then access only the
submission. Both modes keep the same checklist review and acceptance protocol.
Choose per run:

```bash
bash scripts/run_experiment.sh stochastic-interpolants \
  plans/diagnostic/stochastic-interpolants-rubric-assisted/plan.json embedded
bash scripts/run_experiment.sh stochastic-interpolants \
  plans/diagnostic/stochastic-interpolants-rubric-assisted/plan.json tools
```

The same selection is available with `scripts/resume_experiment.sh ...
--checker-materials embedded` or the `RESEARCH_AGENT_CHECKER_MATERIALS` environment
variable. The direct runner accepts `--checker-materials tools|embedded`.

Run from WSL/Linux with Docker available. The default agent API is Qwen3.7-Flash;
the default judge is USTC DeepSeek-Flash. Put their API keys in `.env` or export
them before launching. The example outline is fixed and has no checklist:

```bash
cd agents/ra-skill-list
bash scripts/setup.sh
export DASHSCOPE_API_KEY=...
export USTC_LLM_API_KEY=...
bash scripts/run_experiment.sh all-in-one plans/all-in-one.plan.json
```

The workflow uses PaperBench's official Code-Dev grading pipeline. It does not include
HostUV/Landlock, predownloaded research resources, paper-specific hints, or
modified grading prompts. An optional GLM-4.7-Flash agent API adapter is kept
entirely inside this standalone project.

To run from an already-complete checklist plan, pass its JSON file as the second
argument. The script validates it before creating the run directory and archives
the exact input as `input_plan.json`. This example plan is a rubric-assisted
diagnostic and should be reported separately from blind benchmark runs:

```bash
bash scripts/run_experiment.sh mechanistic-understanding \
  plans/diagnostic/mechanistic-rubric-assisted/plan.json
```

To continue an interrupted experiment with a fresh agent conversation:

```bash
bash scripts/resume_experiment.sh all-in-one \
  ./experiments/all-in-one/2026-09-29T06-54-21Z-ra-skill-list.E3Pvbv
```

The source may be the experiment root, its `workspace`, or its
`workspace/submission`. The script creates a new, unique run and copies the
submission, complete checklist plan, task statuses, rejection counts, and Skill
selection. Already completed/skipped tasks retain their states; execution starts
at the first uncompleted task. The new conversation receives a continuation
briefing and the current task's last rejection, if available. Historical
conversations, checker logs, process IDs, jobs, and old completion/error markers
are not restored. The existing source is not modified. Docker and public paper
inputs are initialized normally, and the exported submission is automatically
graded when the continuation finishes. Installed dependencies from the old
container are not restored; the agent must prepare them in the new container.

For an arbitrary code directory without adjacent experiment state, provide
`--plan COMPLETE_CHECKLIST_PLAN_JSON`; all tasks then start uncompleted. To
preserve approvals from a separate task-state file, also provide
`--state PLAN_STATE_JSON`. Status IDs must exactly match the plan. Supplying a
different plan does not automatically inherit the old plan's approvals.

Add `--prepare-only` to copy and validate a continuation without any model calls,
Docker startup, or grading. It prints the new run root. The prepared directory
can later be executed with `scripts/run_agent.sh PAPER_ID RUN_ROOT
RUN_ROOT/input_plan.json`, followed by `scripts/evaluate_submission.sh`.

To use the optional official Zhipu GLM-4.7-Flash agent configuration while
leaving the PaperBench judge unchanged:

```bash
export RESEARCH_AGENT_API_CONFIG=glm47flash
export ZAI_API_KEY=...
export GRADER_OPENAI_API_KEY=...
bash scripts/run_experiment.sh all-in-one plans/all-in-one.plan.json
```

This selects model `glm-4.7-flash` at the official Zhipu endpoint. Override the
model only when deliberately testing another compatible deployment with
`RESEARCH_AGENT_MODEL`.

To use USTC DeepSeek-Flash for the Planner, Main Agent, and Check Agent:

```bash
export RESEARCH_AGENT_API_CONFIG=ustc-deepseek-flash
export USTC_LLM_API_KEY=...
export GRADER_OPENAI_API_KEY=...
bash scripts/run_experiment.sh all-in-one plans/all-in-one.plan.json
```

This selects the currently available model ID `deepseek-flash` at
`https://api.llm.ustc.edu.cn/v1` with its documented 1M context window.

To use Alibaba Cloud Qwen3.7-Flash with thinking disabled:

```bash
export RESEARCH_AGENT_API_CONFIG=qwen37flash
export DASHSCOPE_API_KEY=...
export GRADER_OPENAI_API_KEY=...
bash scripts/run_experiment.sh all-in-one plans/all-in-one.plan.json
```

This selects `qwen3.7-flash` through the Beijing OpenAI-compatible DashScope
endpoint. The adapter explicitly sends `enable_thinking=false`; the PaperBench
judge remains unchanged unless separately configured.

The evaluator uses USTC DeepSeek-Flash by default. To route all
judge calls (including integer/float structured helper calls) through USTC's
DeepSeek-Flash without changing PaperBench's prompts or scoring logic:

```bash
export PAPERBENCH_GRADER_API_CONFIG=ustc-deepseek-flash
export USTC_LLM_API_KEY=...
bash scripts/evaluate_submission.sh pinn /path/to/submission
```
