"""ReWOO: plan every tool call, run the tools, then solve.

The planner emits ``#E1 = tool[argument]`` lines. Arguments may mention earlier
evidence (``population of #E1``). The worker substitutes those references and
calls tools with no further model turns. The solver writes the answer from the
evidence. That is the token saving versus ReAct and plan-and-execute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.messages import Message, MessageRole
from ..llm.meter import MeteredLLM
from ..tools.registry import ToolRegistry
from .graph import graph_types
from .plan_execute import control_registry, default_control_llm
from .tasks import ControlTask

_LINE = re.compile(r"^#(E\d+)\s*=\s*([A-Za-z_][\w]*)\[(.*)\]\s*$")


@dataclass
class ReWOOResult:
    """One ReWOO run."""

    task_id: str
    answer: str
    expected: str
    success: bool
    runtime: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    llm_calls: int
    evidence: dict[str, str] = field(default_factory=dict)
    tool_trace: list[dict] = field(default_factory=list)
    visited: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def parse_rewoo(plan_text: str) -> list[tuple[str, str, str]]:
    """Return ``(evidence_id, tool, argument)`` in plan order."""
    steps: list[tuple[str, str, str]] = []
    for raw in plan_text.splitlines():
        found = _LINE.match(raw.strip())
        if found:
            steps.append((found.group(1), found.group(2), found.group(3)))
    if not steps:
        raise ValueError(f"ReWOO plan has no #E steps: {plan_text!r}")
    return steps


def substitute(text: str, evidence: dict[str, str]) -> str:
    """Replace ``#E1`` style references with evidence already produced."""
    for key in sorted(evidence, key=len, reverse=True):
        text = text.replace(f"#{key}", evidence[key])
    return text


def run_rewoo(
    task: ControlTask,
    llm: Any = None,
    registry: ToolRegistry | None = None,
    *,
    backend: str = "builtin",
    max_steps: int = 10,
) -> ReWOOResult:
    """Plan, execute every tool call without the model, then solve."""
    inner = llm if llm is not None else default_control_llm()
    meter = MeteredLLM(inner)
    tools = registry if registry is not None else control_registry()
    graph_cls, start, end, runtime = graph_types(backend)
    compiled = _compile(graph_cls, start, end, meter, tools)
    state = compiled.invoke(
        {
            "task_id": task.id,
            "question": task.question,
            "plan_text": "",
            "evidence": {},
            "response": "",
            "tool_trace": [],
            "visited": [],
        },
        config={"recursion_limit": max_steps},
    )
    answer = str(state.get("response") or "").strip()
    return ReWOOResult(
        task_id=task.id,
        answer=answer,
        expected=task.answer,
        success=answer == task.answer,
        runtime=runtime,
        prompt_tokens=meter.prompt_tokens,
        completion_tokens=meter.completion_tokens,
        latency_ms=meter.latency_ms,
        llm_calls=meter.calls,
        evidence=dict(state.get("evidence") or {}),
        tool_trace=list(state.get("tool_trace") or []),
        visited=list(state.get("visited") or []),
    )


def _compile(graph_cls: Any, start: Any, end: Any, meter: MeteredLLM, tools: ToolRegistry) -> Any:
    from typing import TypedDict

    class ReWOOState(TypedDict, total=False):
        task_id: str
        question: str
        plan_text: str
        evidence: dict
        response: str
        tool_trace: list
        visited: list[str]

    def planner(state: dict) -> dict:
        response = meter.complete(
            [
                Message(
                    MessageRole.SYSTEM,
                    "STAGE: rewoo_planner\n"
                    f"TASK: {state['task_id']}\n"
                    "Write a ReWOO plan. One step per line: #E1 = search[query] "
                    "or #E1 = calculator[expression]. Later steps may use #E1.",
                ),
                Message(MessageRole.USER, state["question"]),
            ]
        )
        return {"plan_text": response.content}

    def worker(state: dict) -> dict:
        evidence: dict[str, str] = {}
        trace: list[dict] = []
        for evidence_id, tool_name, argument in parse_rewoo(state["plan_text"]):
            concrete = substitute(argument, evidence)
            if tool_name == "search":
                arguments: dict[str, str] = {"query": concrete}
            elif tool_name == "calculator":
                arguments = {"expression": concrete}
            else:
                raise ValueError(f"ReWOO plan used unknown tool {tool_name!r}")
            result = str(tools.call(tool_name, arguments))
            evidence[evidence_id] = result
            trace.append(
                {
                    "id": evidence_id,
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result,
                }
            )
        return {"evidence": evidence, "tool_trace": trace}

    def solver(state: dict) -> dict:
        evidence = state.get("evidence") or {}
        rendered = "\n".join(f"#{key} = {value}" for key, value in evidence.items())
        response = meter.complete(
            [
                Message(
                    MessageRole.SYSTEM,
                    "STAGE: rewoo_solver\n"
                    f"TASK: {state['task_id']}\n"
                    "Answer from the evidence only. Return the value and nothing else.",
                ),
                Message(
                    MessageRole.USER,
                    f"Question: {state['question']}\nEvidence:\n{rendered}",
                ),
            ]
        )
        return {"response": response.content.strip()}

    graph = graph_cls(ReWOOState)
    graph.add_node("planner", _track("planner", planner))
    graph.add_node("worker", _track("worker", worker))
    graph.add_node("solver", _track("solver", solver))
    graph.add_edge(start, "planner")
    graph.add_edge("planner", "worker")
    graph.add_edge("worker", "solver")
    graph.add_edge("solver", end)
    return graph.compile()


def _track(name: str, fn: Any) -> Any:
    def inner(state: dict) -> dict:
        update = fn(state) or {}
        seen = list(state.get("visited") or [])
        seen.append(name)
        update["visited"] = seen
        return update

    return inner
