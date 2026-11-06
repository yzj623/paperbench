"""Run only the real legacy-outline checklist planner, then stop."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .api_settings import CompatibleAPISettings
from .plan_enrichment import (
    is_legacy_outline,
    parse_legacy_outline,
)
from .plan_enrichment_agent import enrich_legacy_outline_agent
from .qwen37flash_completer import Qwen37FlashTurnCompleterConfig
from .deepseekv4flash_completer import DeepSeekV4FlashTurnCompleterConfig
from .storage import PlanStore


async def run(
    *, workspace: Path,
    outline_path: Path,
    run_dir: Path,
    model: str = "qwen3.7-flash",
    max_steps: int = 256,
) -> Path:
    workspace = workspace.resolve(strict=True)
    required = [
        workspace / "paper" / "paper.md",
        workspace / "paper" / "addendum.md",
        workspace / "instructions.txt",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing public planner input: {path}")
    provider = os.environ.get("RESEARCH_AGENT_API_CONFIG", os.environ.get("AGENT_API_CONFIG", "qwen37flash"))
    if provider not in {"qwen37flash", "deepseekv4flash"}:
        raise ValueError(f"Unsupported planner test API config: {provider}")
    key_env = "DEEPSEEK_API_KEY" if provider == "deepseekv4flash" else "DASHSCOPE_API_KEY"
    if not os.environ.get(key_env):
        raise RuntimeError(f"{key_env} is required for the planner test")

    raw_outline = json.loads(outline_path.read_text(encoding="utf-8"))
    if not is_legacy_outline(raw_outline):
        raise ValueError("This test requires an outline whose every subtask lacks checklist")
    outline = parse_legacy_outline(raw_outline)

    store = PlanStore(run_dir)
    if (store.plan_path.exists() or store.state_path.exists()) and not (
        store.run_dir / "checklist_enrichment_state.json"
    ).exists():
        raise FileExistsError(f"Planner test run is already initialized: {run_dir}")
    api_settings = CompatibleAPISettings.from_config(
        DeepSeekV4FlashTurnCompleterConfig(model="deepseek-v4-flash" if model == "qwen3.7-flash" else model)
        if provider == "deepseekv4flash" else Qwen37FlashTurnCompleterConfig(model=model)
    )
    complete_plan = await enrich_legacy_outline_agent(
        config=api_settings,
        outline=outline,
        workspace_root=workspace,
        store=store,
        max_steps=max_steps,
    )
    item_count = sum(len(task.checklist) for task in complete_plan.subtasks)
    saved = store.load_plan()
    if saved != complete_plan:
        raise RuntimeError("Saved plan does not match the validated enriched plan")
    store.events.append(
        "planner_only_checklist_test_complete",
        subtask_count=len(saved.subtasks),
        checklist_item_count=item_count,
    )
    print(f"Validated checklist plan: {store.plan_path}")
    print(f"Subtasks: {len(saved.subtasks)}")
    print(f"Checklist items: {item_count}")
    print("Planner-only test stopped; no execution, checker, or grading was started.")
    return store.plan_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--outline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.7-flash")
    parser.add_argument("--max-steps", type=int, default=256)
    args = parser.parse_args()
    asyncio.run(run(
        workspace=args.workspace,
        outline_path=args.outline,
        run_dir=args.run_dir,
        model=args.model,
        max_steps=args.max_steps,
    ))


if __name__ == "__main__":
    main()
