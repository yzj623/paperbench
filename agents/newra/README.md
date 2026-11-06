# Standalone ResearchAgent for PaperBench

This project keeps the ResearchAgent implementation outside the official
PaperBench checkout. The only boundary is the benchmark's public task input
and canonical submission directory:

1. `scripts/run_agent.sh PAPER_ID` hydrates only that paper's official Git LFS
   input, creates an isolated `pb-env` Docker workspace, and runs ResearchAgent.
2. The rollout exports a standard `submission/` directory.
3. `scripts/evaluate_submission.sh PAPER_ID SUBMISSION_DIR` passes that folder
   to PaperBench's unmodified `PBDirectSubmissionSolver` and Code-Dev judge.
4. `scripts/run_experiment.sh PAPER_ID` runs both stages in order.

The official checkout is expected at `./PaperBench-Official`, this project
at `./agents/newra`, and outputs at `./experiments`.
Override them with `PAPERBENCH_ROOT`, `RESEARCH_AGENT_ROOT`, and
`EXPERIMENTS_ROOT`.

Run from WSL/Linux with Docker available:

```bash
cd agents/newra
bash scripts/setup.sh
export OPENAI_API_KEY=...
export GRADER_OPENAI_API_KEY=...
bash scripts/run_experiment.sh pinn
```

The default configuration intentionally uses PaperBench's official OpenAI
Responses completer and official judge configuration. It does not include
HostUV/Landlock, predownloaded research resources, paper-specific hints, or
modified grading prompts. An optional GLM-4.7-Flash agent API adapter is kept
entirely inside this standalone project.

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

The evaluator remains on PaperBench's OpenAI judge by default. To route all
judge calls (including integer/float structured helper calls) through USTC's
DeepSeek-Flash without changing PaperBench's prompts or scoring logic:

```bash
export PAPERBENCH_GRADER_API_CONFIG=ustc-deepseek-flash
export USTC_LLM_API_KEY=...
bash scripts/evaluate_submission.sh pinn /path/to/submission
```
