"""Task-success checks over trajectories, states, and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.state import TaskState
    from .recorder import TrajectoryRecorder


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def check_contains(output: str, keywords: list[str]) -> CheckResult:
    missing = [kw for kw in keywords if kw not in output]
    return CheckResult(
        name="contains_keywords",
        passed=not missing,
        detail=f"missing: {missing}" if missing else "all keywords present",
    )


def check_all_subtasks_done(outputs: dict[str, str]) -> CheckResult:
    empty = [k for k, v in outputs.items() if not v or not v.strip()]
    return CheckResult(
        name="all_subtasks_done",
        passed=not empty,
        detail=f"empty outputs: {empty}" if empty else "all subtasks produced output",
    )


def check_no_errors(state: TaskState) -> CheckResult:
    return CheckResult(
        name="no_errors",
        passed=state.ok,
        detail="; ".join(state.errors) if state.errors else "no errors recorded",
    )


def check_step_budget(recorder: TrajectoryRecorder, max_steps: int) -> CheckResult:
    count = len(recorder)
    return CheckResult(
        name="step_budget",
        passed=count <= max_steps,
        detail=f"{count} events against budget of {max_steps}",
    )


def check_agents_involved(
    recorder: TrajectoryRecorder, expected: list[str]
) -> CheckResult:
    seen = set(recorder.agents())
    missing = [a for a in expected if a not in seen]
    return CheckResult(
        name="agents_involved",
        passed=not missing,
        detail=f"missing agents: {missing}" if missing else "all expected agents seen",
    )
