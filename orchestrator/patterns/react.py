"""ReAct baseline for the control-pattern comparison.

This is the existing budgeted tool loop (:class:`ToolCallingLoop`): the model
emits one tool call, sees the observation, and repeats until it answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.agent import Agent
from ..eval.recorder import TrajectoryRecorder
from ..llm.meter import MeteredLLM
from ..tools.loop import MaxIterationsError, ToolCallingLoop
from ..tools.registry import ToolRegistry
from .plan_execute import control_registry, default_control_llm
from .tasks import ControlTask


@dataclass
class ReactResult:
    """One ReAct run on a control task."""

    task_id: str
    answer: str
    expected: str
    success: bool
    runtime: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    llm_calls: int
    error: str = ""
    tool_trace: list[dict] = field(default_factory=list)
    trajectory: TrajectoryRecorder | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def run_react(
    task: ControlTask,
    llm: Any = None,
    registry: ToolRegistry | None = None,
    *,
    max_iterations: int = 8,
) -> ReactResult:
    """Run the budgeted ReAct loop on ``task``."""
    inner = llm if llm is not None else default_control_llm()
    meter = MeteredLLM(inner)
    tools = registry if registry is not None else control_registry()
    agent = Agent(
        name="react",
        role="ReAct agent",
        llm=meter,
        tools=[tools.get("search"), tools.get("calculator")],
        system_prompt=(
            "STAGE: react\n"
            f"TASK: {task.id}\n"
            "You are a ReAct agent. Call one tool at a time. "
            "When the observations are enough, answer with the value only."
        ),
    )
    recorder = TrajectoryRecorder()
    loop = ToolCallingLoop(agent, tools, max_iterations=max_iterations, recorder=recorder)
    error = ""
    try:
        reply = loop.run(task.question)
        answer = reply.content.strip()
    except MaxIterationsError as exc:
        answer = ""
        error = str(exc)
    trace = [
        {"tool": event.detail.get("tool", ""), "result": event.detail.get("result", "")}
        for event in recorder.filter("tool_result")
    ]
    return ReactResult(
        task_id=task.id,
        answer=answer,
        expected=task.answer,
        success=answer == task.answer and not error,
        runtime="tool_loop",
        prompt_tokens=meter.prompt_tokens,
        completion_tokens=meter.completion_tokens,
        latency_ms=meter.latency_ms,
        llm_calls=meter.calls,
        error=error,
        tool_trace=trace,
        trajectory=recorder,
    )
