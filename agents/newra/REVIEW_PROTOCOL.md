# Full-submission review protocol

`newra` starts with a copied standalone NoChecker implementation contract:
the exact execution prompt, protected plan rendering, tool definitions, tool
results, and submission gates. Review instructions and `respond_to_review`
are not exposed during this phase. `skip_subtask` has the same legacy
rejection-count gate as standalone NoChecker.

The first accepted `final_submit` creates the normal PaperBench archive and
an immutable review snapshot. It returns the same result shape and archive
path as standalone NoChecker. The controller then starts full review in the
**same** main-agent conversation. It activates the review execution prompt
and tools only after this boundary, retaining the conversation and encrypted
reasoning caches. Initial archival does not mark the whole experiment done.

Each review creates a fresh read-only Checker. The Checker sees the complete
plan, the current submission, and the previous rounds' issues and responses.
It submits one list of `issue_id`, `description`, `paper_evidence`, and
`code_evidence`. An empty list ends the experiment. For a nonempty list, the
main agent must call `respond_to_review` once per issue, choosing
`plan_inaccurate`, `checker_inaccurate`, or `valid_issue`, with a reason and
evidence. It then calls `final_submit`, with or without code changes. This
saves another snapshot. Reaching the configured round limit also ends the
experiment. The final PaperBench archive is refreshed after that decision;
automatic scoring uses the final submission.

`RESEARCH_AGENT_REVIEW_ROUNDS` controls the number of Checker reviews; the
default is `2` and `0` runs only the implementation phase. The runner also
accepts `--max-review-rounds`. Successful snapshots and protocol history are
stored under `agent-run/research_agent/full_review/`:

- `state.json`: rounds, issues, main-agent responses, phase, snapshot paths.
- `snapshots/submission-NNN.tar.gz`: exact code snapshot after each successful
  `final_submit`, including the initial implementation.
- `../checker_calls.jsonl`, `../tool_calls.jsonl`, `../events.jsonl`: checker,
  tool, and phase events, respectively.

The normal experiment entry point remains `scripts/run_experiment.sh PAPER_ID`.
Set `RESEARCH_AGENT_PLAN` to an existing plan JSON to use a fixed outline.

Both standalone NoChecker and every newra phase wait and retry indefinitely
on OpenAI subscription-window exhaustion. Each wait lasts at least 300 seconds,
or longer when Retry-After requests it. These waits do not consume the normal
request retry deadline. Other errors retain their existing bounded handling.

`tests/fixtures/nochecker_contract.json` pins the copied NoChecker prompt
hashes, tool schemas, and every plan visibility state. It is local test data;
runtime code does not import or read the other agent repositories.
`tests/test_nochecker_phase_parity.py` verifies the contract and the actual
implementation-to-review transition without an LLM or Docker.
