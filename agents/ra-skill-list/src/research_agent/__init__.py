"""Planner/checker based research-paper reproduction agent.

This package is intentionally independent from ``basicagent`` and
``iterativeagent``. It contains the runnable stateful protocol, prompt layer,
read-only checker, and PaperBench solver integration for the new experiment.
"""

from .async_jobs import ManagedJobManager, ManagedJobRegistry
from .checker import (
    CheckAgentContext,
    CheckerResponseError,
    CheckIssue,
    CheckReport,
    parse_check_report,
    review_with_retries,
)
from .context import MessageWindow, trim_message_units
from .plan import (
    Plan,
    PlannedTask,
    PlanTask,
    PlanValidationError,
    TaskStatus,
    parse_plan,
)
from .session import ResearchAgentSession
from .solver import ResearchAgentSolver
from .state_machine import (
    AgentProcess,
    FinalSubmitResult,
    ResearchAgentState,
    ResearchToolError,
    ResearchToolRuntime,
)

__all__ = [
    "AgentProcess",
    "CheckAgentContext",
    "CheckIssue",
    "CheckReport",
    "CheckerResponseError",
    "FinalSubmitResult",
    "MessageWindow",
    "ManagedJobManager",
    "ManagedJobRegistry",
    "Plan",
    "PlanTask",
    "PlanValidationError",
    "PlannedTask",
    "ResearchAgentSession",
    "ResearchAgentSolver",
    "ResearchAgentState",
    "ResearchToolError",
    "ResearchToolRuntime",
    "TaskStatus",
    "parse_check_report",
    "parse_plan",
    "review_with_retries",
    "trim_message_units",
]
