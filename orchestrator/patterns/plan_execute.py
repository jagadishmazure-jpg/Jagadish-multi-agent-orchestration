"""LangGraph-style plan-and-execute: planner, executor, replanner.

The graph is planner -> executor -> replanner, and the replanner either
returns a new plan or a final answer. This is the standalone pattern. The
replanner is not buried inside another orchestrator.

``backend="builtin"`` compiles the local StateGraph. ``backend="langgraph"``
compiles the same nodes with ``langgraph.graph.StateGraph``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.messages import Message, MessageRole
from ..llm.factory import build_llm
from ..llm.meter import MeteredLLM
from ..tools.builtins import make_calculator_tool
from ..tools.registry import Tool, ToolRegistry
from .control_llm import ControlMockLLM
from .graph import graph_types
from .tasks import ControlTask, lookup_fact

_STEP_PREFIX = re.compile(r"^(?:[-*]|\d+[.)])\s*")


@dataclass
class PlanExecuteResult:
    """One plan-and-execute run."""

    task_id: str
    answer: str
    expected: str
    success: bool
    runtime: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    llm_calls: int
    revisions: int
    visited: list[str] = field(default_factory=list)
    tool_trace: list[dict] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def control_registry() -> ToolRegistry:
    """Search over the fixed fact table plus the safe calculator."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="search",
            description="Look up one fact from the offline table.",
            parameters={"query": str},
            func=lookup_fact,
        )
    )
    registry.register(make_calculator_tool())
    return registry


def parse_plan(text: str) -> list[str]:
    """Parse a numbered ``PLAN:`` block into step strings."""
    steps: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.upper().startswith("PLAN") or line.upper().startswith("FINAL:"):
            continue
        cleaned = _STEP_PREFIX.sub("", line).strip()
        if cleaned:
            steps.append(cleaned)
    if not steps:
        raise ValueError(f"plan had no steps: {text!r}")
    return steps


def run_plan_execute(
    task: ControlTask,
    llm: Any = None,
    registry: ToolRegistry | None = None,
    *,
    backend: str = "builtin",
    max_steps: int = 40,
) -> PlanExecuteResult:
    """Run planner, then alternate executor and replanner until a final answer."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    inner = llm if llm is not None else default_control_llm()
    meter = MeteredLLM(inner)
    tools = registry if registry is not None else control_registry()
    graph_cls, start, end, runtime = graph_types(backend)
    compiled = _compile(graph_cls, start, end, meter, tools)
    state = compiled.invoke(
        {
            "task_id": task.id,
            "question": task.question,
            "plan": [],
            "past_steps": [],
            "response": "",
            "revisions": 0,
            "tool_trace": [],
            "visited": [],
        },
        config={"recursion_limit": max_steps},
    )
    answer = str(state.get("response") or "").strip()
    return PlanExecuteResult(
        task_id=task.id,
        answer=answer,
        expected=task.answer,
        success=answer == task.answer,
        runtime=runtime,
        prompt_tokens=meter.prompt_tokens,
        completion_tokens=meter.completion_tokens,
        latency_ms=meter.latency_ms,
        llm_calls=meter.calls,
        revisions=int(state.get("revisions") or 0),
        visited=list(state.get("visited") or []),
        tool_trace=list(state.get("tool_trace") or []),
        plan=list(state.get("plan") or []),
    )


def default_control_llm() -> Any:
    from ..llm.factory import azure_config

    if azure_config() is not None:
        return build_llm()
    return ControlMockLLM()


def _compile(graph_cls: Any, start: Any, end: Any, meter: MeteredLLM, tools: ToolRegistry) -> Any:
    from typing import TypedDict

    class PlanState(TypedDict, total=False):
        task_id: str
        question: str
        plan: list[str]
        past_steps: list[list[str]]
        response: str
        revisions: int
        tool_trace: list
        visited: list[str]

    def planner(state: dict) -> dict:
        response = meter.complete(
            [
                Message(
                    MessageRole.SYSTEM,
                    "STAGE: planner\n"
                    f"TASK: {state['task_id']}\n"
                    "You are the planner. Return a numbered PLAN. "
                    "Each step is 'search: query' or 'calculator: expression'.",
                ),
                Message(MessageRole.USER, state["question"]),
            ]
        )
        return {"plan": parse_plan(response.content)}

    def executor(state: dict) -> dict:
        step = state["plan"][0]
        response = meter.complete(
            [
                Message(
                    MessageRole.SYSTEM,
                    "STAGE: executor\n"
                    f"TASK: {state['task_id']}\n"
                    f"STEP: {step}\n"
                    "Call exactly one tool for this step.",
                ),
                Message(
                    MessageRole.USER,
                    f"Question: {state['question']}\nExecute this step:\n{step}",
                ),
            ],
            tools=tools.schemas(),
        )
        if response.tool_calls:
            call = response.tool_calls[0]
            result = str(tools.call(call.name, call.arguments))
            entry = {"tool": call.name, "arguments": dict(call.arguments), "result": result}
        else:
            result = response.content or "no tool call"
            entry = {"tool": "", "arguments": {}, "result": result}
        past = [list(item) for item in state.get("past_steps") or []]
        past.append([step, result])
        trace = list(state.get("tool_trace") or [])
        trace.append(entry)
        return {"past_steps": past, "tool_trace": trace}

    def replanner(state: dict) -> dict:
        past = state.get("past_steps") or []
        rendered = "\n".join(
            f"{index}. {step} -> {result}" for index, (step, result) in enumerate(past, start=1)
        )
        current = "\n".join(state.get("plan") or [])
        response = meter.complete(
            [
                Message(
                    MessageRole.SYSTEM,
                    "STAGE: replanner\n"
                    f"TASK: {state['task_id']}\n"
                    f"PAST: {len(past)}\n"
                    "Return PLAN: with the remaining steps, or FINAL: and the answer. "
                    "Add steps the original plan missed.",
                ),
                Message(
                    MessageRole.USER,
                    f"Question: {state['question']}\n"
                    f"Completed steps:\n{rendered}\n"
                    f"Current plan:\n{current}",
                ),
            ]
        )
        text = response.content.strip()
        if text.startswith("FINAL:"):
            return {"response": text.split(":", 1)[1].strip(), "plan": []}
        new_plan = parse_plan(text)
        old_tail = list(state.get("plan") or [])[1:]
        revisions = int(state.get("revisions") or 0)
        if new_plan != old_tail:
            revisions += 1
        return {"plan": new_plan, "revisions": revisions}

    def route(state: dict) -> Any:
        if state.get("response"):
            return end
        if state.get("plan"):
            return "executor"
        return end

    graph = graph_cls(PlanState)
    graph.add_node("planner", _track("planner", planner))
    graph.add_node("executor", _track("executor", executor))
    graph.add_node("replanner", _track("replanner", replanner))
    graph.add_edge(start, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")
    graph.add_conditional_edges("replanner", route, {"executor": "executor", end: end})
    return graph.compile()


def _track(name: str, fn: Any) -> Any:
    def inner(state: dict) -> dict:
        update = fn(state) or {}
        seen = list(state.get("visited") or [])
        seen.append(name)
        update["visited"] = seen
        return update

    return inner
