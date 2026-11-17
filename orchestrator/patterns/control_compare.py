"""Compare ReAct, plan-and-execute, and ReWOO on the same tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .plan_execute import PlanExecuteResult, run_plan_execute
from .react import ReactResult, run_react
from .rewoo import ReWOOResult, run_rewoo
from .tasks import TASKS, ControlTask

Run = ReactResult | PlanExecuteResult | ReWOOResult


@dataclass
class PatternScore:
    """Totals for one pattern across the task set."""

    pattern: str
    success_rate: float
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    llm_calls: int
    passed: int
    tasks: int
    runtime: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def compare_control(
    llm: Any = None,
    tasks: tuple[ControlTask, ...] | None = None,
    *,
    backend: str = "builtin",
) -> list[PatternScore]:
    """Score ReAct, plan-and-execute, and ReWOO.

    Each task builds its own meter around ``llm``. The scripted model is
    stateless, so one instance can serve every task.
    """
    chosen = tasks if tasks is not None else TASKS
    grouped: dict[str, list[Run]] = {"react": [], "plan_execute": [], "rewoo": []}
    for task in chosen:
        grouped["react"].append(run_react(task, llm=llm))
        grouped["plan_execute"].append(run_plan_execute(task, llm=llm, backend=backend))
        grouped["rewoo"].append(run_rewoo(task, llm=llm, backend=backend))
    return [_score(name, runs) for name, runs in grouped.items()]


def _score(pattern: str, runs: list[Run]) -> PatternScore:
    passed = sum(1 for run in runs if run.success)
    return PatternScore(
        pattern=pattern,
        success_rate=(passed / len(runs)) if runs else 0.0,
        prompt_tokens=sum(run.prompt_tokens for run in runs),
        completion_tokens=sum(run.completion_tokens for run in runs),
        latency_ms=sum(run.latency_ms for run in runs),
        llm_calls=sum(run.llm_calls for run in runs),
        passed=passed,
        tasks=len(runs),
        runtime=runs[0].runtime if runs else "",
    )
