"""Tool-driven, resumable checklist enrichment for a fixed legacy outline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from openai.types.responses import FunctionToolParam
from paperbench.solvers.basicagent.tools.base import Tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .completer_resilience import make_resilient_completer_request
from .plan import Plan, SourceRef, parse_plan
from .plan_enrichment import (
    CHECKLIST_ENRICHMENT_PROMPT,
    CHECKLIST_ENRICHMENT_REQUEST,
    LegacyOutline,
    public_material_message,
)
from .storage import PlanStore, atomic_json_write


class ChecklistItemInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criterion: str
    sources: list[SourceRef] = Field(min_length=1)

    @field_validator("criterion")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("criterion must be non-empty")
        return value


class SubmitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    checklist: list[ChecklistItemInput]


class SubmitTaskChecklistTool(Tool):
    task_ids: list[str]

    def name(self) -> str:
        return "submit_task_checklist"

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function", name=self.name(), strict=False,
            description=("Submit the complete checklist for exactly one existing subtask. "
                         "An empty checklist is valid and counts as a submission. "
                         "For each subtask, quote existence can reject a submission only once. "
                         "The search ignores all whitespace and letter case; Markdown markers, "
                         "formula delimiters, punctuation, and other characters must match. "
                         "On the first not-found result, inspect the source and decide whether "
                         "to correct it. Later submissions for that subtask skip quote-existence "
                         "checks; acceptance does not verify their quotes. "
                         "A later successful call for the same task replaces the old checklist. "
                         "Invalid calls leave the prior checklist unchanged."),
            parameters={
                "type": "object", "additionalProperties": False,
                "properties": {
                    "task_id": {"type": "string", "enum": self.task_ids},
                    "checklist": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "criterion": {"type": "string", "minLength": 1},
                            "sources": {"type": "array", "minItems": 1, "items": {
                                "type": "object", "additionalProperties": False,
                                "properties": {
                                    "document": {"type": "string", "enum": ["paper.md", "addendum.md", "instructions.txt"]},
                                    "quote": {"type": "string", "minLength": 1},
                                },
                                "required": ["document", "quote"],
                            }},
                        },
                        "required": ["criterion", "sources"],
                    }},
                },
                "required": ["task_id", "checklist"],
            },
        )

    async def execute(self, computer: Any, **kwargs: Any) -> str:
        raise RuntimeError("submit_task_checklist is intercepted by the enrichment agent")


class FinishPlanTool(Tool):
    def name(self) -> str:
        return "finish_plan"

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function", name=self.name(), strict=False,
            description=("Finish and save the fixed outline with its submitted checklists. "
                         "Only succeeds after every task has a successful submission; "
                         "otherwise returns all missing task IDs. Call with an empty object."),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
        )

    async def execute(self, computer: Any, **kwargs: Any) -> str:
        raise RuntimeError("finish_plan is intercepted by the enrichment agent")


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"ok": False, "code": code, "message": message, "details": details}


def _normalized_quote(text: str) -> str:
    """Allow whitespace and case differences, but preserve every other character."""

    return re.sub(r"\s+", "", text).casefold()


def _unmatched_sources(
    item: SubmitInput, normalized_documents: dict[str, str], documents: dict[str, str]
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for item_index, checklist_item in enumerate(item.checklist, 1):
        for source_index, source in enumerate(checklist_item.sources, 1):
            if _normalized_quote(source.quote) not in normalized_documents[source.document]:
                detail: dict[str, Any] = {
                    "item_index": item_index,
                    "source_index": source_index,
                    "document": source.document,
                }
                matches_elsewhere = [
                    name for name, content in normalized_documents.items()
                    if name != source.document and _normalized_quote(source.quote) in content
                ]
                if matches_elsewhere:
                    detail["quote_found_in_other_documents"] = matches_elsewhere
                else:
                    for line in documents[source.document].splitlines():
                        if _normalized_quote(source.quote) in _normalized_quote(line.replace("`", "")):
                            detail["nearby_document_line"] = line[:500]
                            break
                missing.append(detail)
    return missing


def _build_plan(outline: LegacyOutline, submitted: dict[str, list[dict[str, Any]]]) -> Plan:
    return parse_plan({"subtasks": [
        {**task.model_dump(), "checklist": [
            {"id": f"{task.id}_check_{index:03d}", **item}
            for index, item in enumerate(submitted[task.id], 1)
        ]}
        for task in outline.subtasks
    ]})


async def enrich_legacy_outline_agent(
    *, config: Any, outline: LegacyOutline, workspace_root: Path, store: PlanStore,
    max_steps: int = 256,
) -> Plan:
    """Continue an enrichment conversation until finish_plan freezes the plan."""

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    initial = [
        {"role": "system", "content": CHECKLIST_ENRICHMENT_PROMPT},
        {"role": "user", "content": public_material_message(workspace_root, outline)},
        {"role": "user", "content": CHECKLIST_ENRICHMENT_REQUEST},
    ]
    digest = hashlib.sha256(json.dumps(initial, ensure_ascii=False).encode()).hexdigest()
    state_path = store.run_dir / "checklist_enrichment_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("input_digest") != digest:
            raise ValueError("Cannot resume checklist enrichment: outline, public materials, or prompt changed")
        if store.plan_path.exists():
            return store.load_plan()
        messages = state["messages"]
        submitted = state["submitted"]
        steps = state["steps"]
        quote_rejection_used = set(state.get("quote_rejection_used_task_ids", []))
    else:
        if store.plan_path.exists() or store.state_path.exists():
            raise FileExistsError(f"Plan is already initialized: {store.run_dir}")
        messages = initial
        submitted = {}
        steps = 0
        quote_rejection_used: set[str] = set()
        atomic_json_write(state_path, {
            "input_digest": digest, "messages": messages, "submitted": submitted, "steps": steps,
            "quote_rejection_used_task_ids": [],
        })

    task_ids = [task.id for task in outline.subtasks]
    root = workspace_root.resolve(strict=True)
    documents = {
        name: path.read_text(encoding="utf-8")
        for name, path in (
            ("paper.md", root / "paper" / "paper.md"),
            ("addendum.md", root / "paper" / "addendum.md"),
            ("instructions.txt", root / "instructions.txt"),
        )
    }
    normalized_documents = {
        name: _normalized_quote(content) for name, content in documents.items()
    }
    tools = [SubmitTaskChecklistTool(task_ids=task_ids), FinishPlanTool()]
    completer = config.new_config(basicagent_tools=tools).build()

    stop_after = steps + max_steps
    while steps < stop_after:
        response = await make_resilient_completer_request(completer=completer, messages=messages)
        messages.extend(response.response_messages)
        calls = response.tool_calls
        if not calls:
            missing = [task_id for task_id in task_ids if task_id not in submitted]
            messages.append({"role": "user", "content": (
                "You must call submit_task_checklist for each remaining task, then finish_plan. "
                f"Missing task IDs: {missing}. A plain-text response does not finish the plan."
            )})
            store.events.append("checklist_enrichment_no_tool", missing_task_ids=missing)
        else:
            finished = False
            for index, call in enumerate(calls):
                if index:
                    result = _error(
                        "MULTIPLE_TOOL_CALLS_SKIPPED",
                        "This response called multiple tools; only the first was executed. "
                        "This tool was not executed. Wait for the results and call it again next round.",
                    )
                elif call.name == "submit_task_checklist":
                    try:
                        arguments = dict(call.arguments)
                        if isinstance(arguments.get("checklist"), str):
                            try:
                                arguments["checklist"] = json.loads(arguments["checklist"])
                            except json.JSONDecodeError as exc:
                                result = _error(
                                    "INVALID_CHECKLIST",
                                    "checklist must be a JSON array, not a string containing malformed JSON. "
                                    "Pass a native array in the tool arguments; do not quote the whole array. "
                                    "If text inside an item contains quotation marks, escape them correctly.",
                                    json_error=f"{exc.msg} at position {exc.pos}",
                                )
                                messages.append({"role": "tool", "tool_call_id": call.call_id,
                                                 "content": json.dumps(result, ensure_ascii=False)})
                                store.tool_log.append("checklist_enrichment_tool", tool=call.name,
                                                      arguments=call.arguments, result=result)
                                continue
                        item = SubmitInput.model_validate(arguments)
                        if item.task_id not in task_ids:
                            result = _error("UNKNOWN_TASK_ID", "Task ID is not in the fixed outline", task_id=item.task_id)
                        else:
                            quote_check_performed = item.task_id not in quote_rejection_used
                            unmatched = (
                                _unmatched_sources(item, normalized_documents, documents)
                                if quote_check_performed else []
                            )
                            if unmatched:
                                quote_rejection_used.add(item.task_id)
                                result = _error(
                                    "QUOTE_NOT_FOUND",
                                    "This is the only quote-existence rejection for this subtask. "
                                    "These quotes were not found by a continuous-text search in "
                                    "their named documents. The search ignores whitespace and "
                                    "letter case, but not Markdown markers, formula delimiters, "
                                    "or punctuation. Decide yourself whether to correct the quotes "
                                    "before resubmitting. Future submissions for this subtask will "
                                    "skip quote-existence checks; acceptance will not verify them. "
                                    "This call did not save a checklist.",
                                    task_id=item.task_id,
                                    unmatched_sources=unmatched,
                                )
                            else:
                                submitted[item.task_id] = [entry.model_dump() for entry in item.checklist]
                                result = {"ok": True, "task_id": item.task_id,
                                          "item_count": len(item.checklist),
                                          "quote_existence_checked": quote_check_performed,
                                          "remaining_task_ids": [
                                              task_id for task_id in task_ids if task_id not in submitted
                                          ]}
                    except Exception as exc:
                        result = _error("INVALID_CHECKLIST", "Invalid checklist submission; no state changed", error=str(exc))
                elif call.name == "finish_plan":
                    if call.arguments:
                        result = _error("INVALID_FINISH_ARGUMENTS", "finish_plan requires an empty object")
                    else:
                        missing = [task_id for task_id in task_ids if task_id not in submitted]
                        if missing:
                            result = _error("MISSING_TASKS", "Submit checklists for all missing task IDs first", missing_task_ids=missing)
                        else:
                            try:
                                plan = _build_plan(outline, submitted)
                                store.initialize(plan)
                                result = {"ok": True, "plan_path": str(store.plan_path)}
                                finished = True
                            except Exception as exc:
                                result = _error("FINALIZATION_FAILED", "Could not validate or save the final plan", error=str(exc))
                else:
                    result = _error("UNKNOWN_TOOL", "Tool is unavailable", tool=call.name)
                messages.append({"role": "tool", "tool_call_id": call.call_id,
                                 "content": json.dumps(result, ensure_ascii=False)})
                store.tool_log.append("checklist_enrichment_tool", tool=call.name,
                                      arguments=call.arguments, result=result)
            if finished:
                steps += 1
                atomic_json_write(state_path, {"input_digest": digest, "messages": messages,
                                               "submitted": submitted, "steps": steps, "finished": True,
                                               "quote_rejection_used_task_ids": sorted(quote_rejection_used)})
                return store.load_plan()
        steps += 1
        atomic_json_write(state_path, {"input_digest": digest, "messages": messages,
                                       "submitted": submitted, "steps": steps,
                                       "quote_rejection_used_task_ids": sorted(quote_rejection_used)})
    raise RuntimeError(f"Checklist enrichment reached the {max_steps}-step limit for this run; resume state is saved at {state_path}")
