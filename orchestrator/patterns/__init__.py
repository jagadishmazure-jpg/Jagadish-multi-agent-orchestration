"""Five orchestration patterns: supervisor, router, sequential, blackboard, hierarchical."""

from .blackboard import Blackboard, BlackboardResult
from .hierarchical import Hierarchical, HierarchicalResult, UnknownTeamError
from .router import Router, RouterResult, UnknownRouteError
from .sequential import PipelineResult, SequentialPipeline
from .supervisor import (
    Supervisor,
    SupervisorResult,
    UnknownWorkerError,
    parse_subtasks,
)

__all__ = [
    "Blackboard",
    "BlackboardResult",
    "Hierarchical",
    "HierarchicalResult",
    "PipelineResult",
    "Router",
    "RouterResult",
    "SequentialPipeline",
    "Supervisor",
    "SupervisorResult",
    "UnknownRouteError",
    "UnknownTeamError",
    "UnknownWorkerError",
    "parse_subtasks",
]
