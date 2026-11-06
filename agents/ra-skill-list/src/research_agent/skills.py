"""Per-subtask skill catalog, selection, persistence, and protected-message rendering."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from openai.types.responses import FunctionToolParam
from paperbench.solvers.basicagent.tools.base import Tool

from .storage import PlanStore, atomic_json_write


MAX_SELECTED_SKILLS = 4
MAX_SELECTION_ATTEMPTS = 3
MAX_SELECTOR_STEPS = 64


@dataclass(frozen=True)
class SkillDocument:
    skill_id: str
    name: str
    description: str
    relative_path: str
    content: str
    content_hash: str

    def metadata(self) -> dict[str, str]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "relative_path": self.relative_path,
        }


class SkillCatalog:
    def __init__(self, *, root: Path, target: str):
        self.root = root.resolve()
        self.target = target
        self.target_root = (self.root / target).resolve()
        if self.root not in self.target_root.parents:
            raise ValueError(f"Invalid skill target: {target!r}")
        if not self.target_root.is_dir():
            raise FileNotFoundError(f"Skill target not found: {self.target_root}")
        self.documents = self._load_documents()
        if not self.documents:
            raise FileNotFoundError(f"No SKILL.md files found for target: {target}")
        digest = hashlib.sha256()
        for skill_id, document in sorted(self.documents.items()):
            digest.update(skill_id.encode("utf-8"))
            digest.update(document.content_hash.encode("ascii"))
        self.version = digest.hexdigest()

    @staticmethod
    def _frontmatter_value(content: str, key: str) -> str | None:
        if not content.startswith("---\n"):
            return None
        end = content.find("\n---", 4)
        if end < 0:
            return None
        pattern = rf"(?mi)^{re.escape(key)}:\s*[\"']?(.*?)[\"']?\s*$"
        match = re.search(pattern, content[4:end])
        return match.group(1).strip() if match else None

    @staticmethod
    def _title(content: str, fallback: str) -> str:
        frontmatter = SkillCatalog._frontmatter_value(content, "name")
        if frontmatter:
            return frontmatter
        match = re.search(r"(?m)^#\s+(.+?)\s*$", content)
        return match.group(1).strip() if match else fallback.replace("-", " ").replace("_", " ")

    @staticmethod
    def _description(content: str) -> str:
        frontmatter = SkillCatalog._frontmatter_value(content, "description")
        if frontmatter:
            return frontmatter
        body = re.sub(r"\A---\n.*?\n---\s*", "", content, flags=re.DOTALL)
        paragraphs = [
            re.sub(r"\s+", " ", item).strip()
            for item in re.split(r"\n\s*\n", body)
            if item.strip() and not item.lstrip().startswith("#")
        ]
        return (paragraphs[0] if paragraphs else "No description supplied.")[:800]

    def _load_documents(self) -> dict[str, SkillDocument]:
        documents: dict[str, SkillDocument] = {}
        for path in sorted(self.target_root.rglob("SKILL.md")):
            relative = path.relative_to(self.target_root)
            skill_id = relative.parent.as_posix()
            if skill_id == ".":
                skill_id = self.target
            if skill_id in documents:
                raise ValueError(f"Duplicate skill ID: {skill_id}")
            content = path.read_text(encoding="utf-8")
            documents[skill_id] = SkillDocument(
                skill_id=skill_id,
                name=self._title(content, relative.parent.name or self.target),
                description=self._description(content),
                relative_path=relative.as_posix(),
                content=content,
                content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        return documents

    def metadata(self) -> list[dict[str, str]]:
        return [self.documents[key].metadata() for key in sorted(self.documents)]


class SelectSkillsTool(Tool):
    catalog: Any
    task_ids: Any
    selected: Any = None
    selection_attempts: int = 0

    class Config:
        arbitrary_types_allowed = True

    def name(self) -> str:
        return "select_skills"

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description=(
                "Select zero to four skills for every subtask in the frozen plan. "
                "Use exact task and skill IDs and give a concise reason for each skill."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_selections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task_id": {"type": "string"},
                                "skills": {
                                    "type": "array",
                                    "minItems": 0,
                                    "maxItems": MAX_SELECTED_SKILLS,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "skill_id": {"type": "string"},
                                            "reason": {"type": "string"},
                                        },
                                        "required": ["skill_id", "reason"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["task_id", "skills"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["task_selections"],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, computer: Any, **kwargs: Any) -> str:
        del computer
        self.selection_attempts += 1
        errors: list[str] = []
        task_selections = kwargs.get("task_selections")
        expected = list(self.task_ids)
        expected_set = set(expected)
        normalized: dict[str, list[dict[str, str]]] = {}
        if not isinstance(task_selections, list):
            errors.append("'task_selections' must be an array.")
        else:
            seen_tasks: set[str] = set()
            for index, item in enumerate(task_selections):
                if not isinstance(item, dict):
                    errors.append(f"Task selection {index + 1} must be an object.")
                    continue
                task_id = item.get("task_id")
                skills = item.get("skills")
                if not isinstance(task_id, str) or not task_id:
                    errors.append(f"Task selection {index + 1} has no valid task_id.")
                    continue
                if task_id not in expected_set:
                    errors.append(f"Unknown task ID: {task_id}")
                if task_id in seen_tasks:
                    errors.append(f"Duplicate task ID: {task_id}")
                seen_tasks.add(task_id)
                if not isinstance(skills, list):
                    errors.append(f"Skills for task {task_id} must be an array.")
                    continue
                if len(skills) > MAX_SELECTED_SKILLS:
                    errors.append(
                        f"At most {MAX_SELECTED_SKILLS} skills may be selected for task {task_id}."
                    )
                normalized_skills: list[dict[str, str]] = []
                seen_skills: set[str] = set()
                for skill_index, skill in enumerate(skills):
                    if not isinstance(skill, dict):
                        errors.append(
                            f"Skill {skill_index + 1} for task {task_id} must be an object."
                        )
                        continue
                    skill_id = skill.get("skill_id")
                    reason = skill.get("reason")
                    if not isinstance(skill_id, str) or not skill_id:
                        errors.append(
                            f"Skill {skill_index + 1} for task {task_id} has no valid skill_id."
                        )
                        continue
                    if skill_id not in self.catalog.documents:
                        errors.append(f"Unknown skill ID: {skill_id}")
                    if skill_id in seen_skills:
                        errors.append(f"Duplicate skill ID {skill_id} for task {task_id}")
                    seen_skills.add(skill_id)
                    if not isinstance(reason, str) or not reason.strip():
                        errors.append(
                            f"Skill {skill_index + 1} for task {task_id} has no non-empty reason."
                        )
                    normalized_skills.append(
                        {"skill_id": skill_id, "reason": str(reason).strip()}
                    )
                normalized[task_id] = normalized_skills
            missing = [task_id for task_id in expected if task_id not in seen_tasks]
            if missing:
                errors.append(f"Missing task IDs: {', '.join(missing)}")
        if errors:
            return json.dumps(
                {
                    "ok": False,
                    "error": "INVALID_SKILL_SELECTION",
                    "messages": errors,
                    "attempt": self.selection_attempts,
                    "maximum_attempts": MAX_SELECTION_ATTEMPTS,
                    "instruction": (
                        "Correct every error and call select_skills again with exactly one "
                        "entry for every task in the frozen plan."
                    ),
                },
                ensure_ascii=False,
            )
        self.selected = normalized
        return json.dumps({"ok": True, "task_selections": normalized}, ensure_ascii=False)


class SkillSelector:
    def __init__(
        self,
        *,
        catalog: SkillCatalog,
        config: Any,
        read_only_tools: Sequence[Any],
        computer: Any,
        store: PlanStore,
        workspace_path: str,
    ):
        self.catalog = catalog
        self.config = config
        self.read_only_tools = list(read_only_tools)
        self.computer = computer
        self.store = store
        self.workspace_path = workspace_path

    def _system_prompt(self) -> str:
        full_plan = json.dumps(
            {"subtasks": self.store.render()}, ensure_ascii=False, indent=2
        )
        catalog = json.dumps(self.catalog.metadata(), ensure_ascii=False, indent=2)
        return f"""You are the Skill Selection Agent for a research reproduction task.

Your only responsibility is to select the skills directly useful for every subtask in the frozen
plan. Do not
implement the subtask, modify files, execute shell commands, revise the plan, or advise the main
agent. You may inspect the paper, addendum, public instructions, and current submission only through
the supplied read-only tools.

For every subtask, select between zero and four skills by making exactly one valid select_skills
tool call. Include every task ID exactly once, using an empty skills array when none adds concrete
value. Prefer the smallest sufficient set, avoid redundant skills, and order each task's selections
from most useful to least useful. Never infer skill contents beyond the metadata. Task sources and
the frozen plan remain authoritative.

Current workspace: {self.workspace_path}
Paper: {self.workspace_path}/paper/paper.md
Addendum: {self.workspace_path}/paper/addendum.md
Public instructions: {self.workspace_path}/instructions.txt
Submission: {self.workspace_path}/submission/

COMPLETE FROZEN PLAN
{full_plan}

SKILL CATALOG
{catalog}
"""

    async def select(self) -> dict[str, list[dict[str, str]]]:
        from .completer_resilience import make_resilient_completer_request

        task_ids = [task.id for task in self.store.load_plan().subtasks]
        selection_tool = SelectSkillsTool(catalog=self.catalog, task_ids=task_ids)
        tools = [*self.read_only_tools, selection_tool]
        by_name = {tool.name(): tool for tool in tools}
        completer = self.config.new_config(basicagent_tools=tools).build()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()}
        ]
        for step in range(MAX_SELECTOR_STEPS):
            response = await make_resilient_completer_request(
                completer=completer, messages=messages
            )
            messages.extend(response.response_messages)
            self.store.skill_log.append(
                "skill_agent_response",
                task_ids=task_ids,
                step=step + 1,
                tool_calls=[call.name for call in response.tool_calls],
            )
            if not response.tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "ERROR: You must finish with one valid select_skills tool call. "
                            "Inspect additional files if needed, then submit all task selections."
                        ),
                    }
                )
                continue
            for call in response.tool_calls:
                tool = by_name.get(call.name)
                if tool is None:
                    output = json.dumps(
                        {
                            "ok": False,
                            "error": "UNSUPPORTED_TOOL",
                            "message": f"Unsupported tool: {call.name}",
                        }
                    )
                else:
                    try:
                        output = await tool.execute(self.computer, **call.arguments)
                    except Exception as exc:
                        output = json.dumps(
                            {
                                "ok": False,
                                "error": type(exc).__name__,
                                "message": str(exc),
                            },
                            ensure_ascii=False,
                        )
                self.store.skill_log.append(
                    "skill_tool_result",
                    task_ids=task_ids,
                    tool=call.name,
                    arguments=dict(call.arguments),
                    result=output,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": str(output),
                    }
                )
                if call.name == "select_skills" and selection_tool.selected is not None:
                    return dict(selection_tool.selected)
                if selection_tool.selection_attempts >= MAX_SELECTION_ATTEMPTS:
                    raise RuntimeError("Skill selection failed after three invalid selections")
        raise RuntimeError("Skill agent exceeded its tool-loop step limit")


class SkillCoordinator:
    def __init__(self, *, catalog: SkillCatalog, selector: SkillSelector, store: PlanStore):
        self.catalog = catalog
        self.selector = selector
        self.store = store
        self.active_task_id: str | None = None

    def _render(self, task_id: str, selections: list[dict[str, str]]) -> str | None:
        if not selections:
            return None
        sections = [
            f'<ACTIVE_SKILLS current_subtask_id="{task_id}">',
            (
                "The following skills are auxiliary reference material for the current subtask. "
                "Task sources and the frozen plan remain authoritative."
            ),
        ]
        for index, selection in enumerate(selections, start=1):
            document = self.catalog.documents[selection["skill_id"]]
            sections.extend(
                [
                    f"--- BEGIN SKILL {index} ---",
                    f"Skill ID: {document.skill_id}",
                    document.content,
                    f"--- END SKILL {index} ---",
                ]
            )
        sections.append("</ACTIVE_SKILLS>")
        return "\n\n".join(sections)

    def _restore(self) -> dict[str, list[dict[str, str]]] | None:
        path = self.store.skill_selection_path
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            task_selections = record["task_selections"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if (
            record.get("target") != self.catalog.target
            or record.get("catalog_version") != self.catalog.version
            or not isinstance(task_selections, dict)
        ):
            return None
        expected = [task.id for task in self.store.load_plan().subtasks]
        if set(task_selections) != set(expected):
            return None
        restored: dict[str, list[dict[str, str]]] = {}
        for task_id in expected:
            items = task_selections.get(task_id)
            if not isinstance(items, list) or len(items) > MAX_SELECTED_SKILLS:
                return None
            restored[task_id] = []
            for item in items:
                if not isinstance(item, dict):
                    return None
                skill_id = item.get("skill_id")
                document = self.catalog.documents.get(skill_id)
                if document is None or item.get("content_hash") != document.content_hash:
                    return None
                restored[task_id].append(
                    {"skill_id": skill_id, "reason": item.get("reason", "")}
                )
        return restored

    def _save(
        self, task_selections: dict[str, list[dict[str, str]]], *, status: str
    ) -> None:
        stored: dict[str, list[dict[str, str]]] = {}
        for task_id, selections in task_selections.items():
            stored[task_id] = []
            for item in selections:
                document = self.catalog.documents[item["skill_id"]]
                stored[task_id].append({**item, "content_hash": document.content_hash})
        atomic_json_write(
            self.store.skill_selection_path,
            {
                "target": self.catalog.target,
                "catalog_version": self.catalog.version,
                "status": status,
                "task_selections": stored,
            },
        )

    async def ensure_current(self, session: Any) -> None:
        if session.window is None:
            raise RuntimeError("execution context is missing")
        task = self.store.first_uncompleted_task()
        if task is None:
            session.window.update_skill(None)
            self.active_task_id = None
            return
        if self.active_task_id == task.id:
            return
        task_selections = self._restore()
        if task_selections is None:
            task_selections = await self.selector.select()
            self._save(task_selections, status="selected")
            status = "skill_plan_selected"
        else:
            status = "skill_plan_restored"
        selections = task_selections[task.id]
        session.window.update_skill(self._render(task.id, selections))
        self.active_task_id = task.id
        self.store.events.append(
            status,
            task_id=task.id,
            selected_skill_ids=[item["skill_id"] for item in selections],
        )

