"""Orchestration patterns: the original five, plus plan-and-execute, ReWOO, and ReAct."""

from .blackboard import Blackboard, BlackboardResult
from .control_compare import PatternScore, compare_control
from .hierarchical import Hierarchical, HierarchicalResult, UnknownTeamError
from .plan_execute import PlanExecuteResult, run_plan_execute
from .react import ReactResult, run_react
from .rewoo import ReWOOResult, run_rewoo
from .router import Router, RouterResult, UnknownRouteError
from .sequential import PipelineResult, SequentialPipeline
from .supervisor import (
    Supervisor,
    SupervisorResult,
    UnknownWorkerError,
    parse_subtasks,
)
from .tasks import TASKS, ControlTask

__all__ = [
    "Blackboard",
    "BlackboardResult",
    "ControlTask",
    "Hierarchical",
    "HierarchicalResult",
    "PatternScore",
    "PipelineResult",
    "PlanExecuteResult",
    "ReactResult",
    "ReWOOResult",
    "Router",
    "RouterResult",
    "SequentialPipeline",
    "Supervisor",
    "SupervisorResult",
    "TASKS",
    "UnknownRouteError",
    "UnknownTeamError",
    "UnknownWorkerError",
    "compare_control",
    "parse_subtasks",
    "run_plan_execute",
    "run_react",
    "run_rewoo",
]
