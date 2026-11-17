"""Scripted model for the ReAct, plan-and-execute, and ReWOO comparison.

Nodes put ``STAGE:`` and ``TASK:`` in the system message. The model returns
the golden script for that stage. It does not infer the answer from tool
output; the pattern code is what runs the tools and threads the results.
"""

from __future__ import annotations

import re

from ..core.messages import MessageRole
from ..llm.client import LLMResponse, ToolCall
from .tasks import ControlTask, task_by_id

_STAGE = re.compile(r"^STAGE:\s*(\S+)\s*$", re.MULTILINE)
_TASK = re.compile(r"^TASK:\s*(\S+)\s*$", re.MULTILINE)
_PAST = re.compile(r"^PAST:\s*(\d+)\s*$", re.MULTILINE)
_STEP = re.compile(r"^STEP:\s*(.+)\s*$", re.MULTILINE)


class ControlMockLLM:
    """Deterministic completions for the multi-step control-pattern tasks."""

    def __init__(self, name: str = "control-mock") -> None:
        self.name = name

    def complete(self, messages: list, tools: list[dict] | None = None) -> LLMResponse:
        blob = "\n".join(message.content for message in messages)
        stage = _required(_STAGE, blob, "STAGE")
        task = task_by_id(_required(_TASK, blob, "TASK"))
        if stage == "planner":
            return self._text(_format_plan(task.initial_plan))
        if stage == "replanner":
            past = int(_required(_PAST, blob, "PAST"))
            try:
                script = task.replans[past]
            except KeyError:
                raise ValueError(f"no replan script for {task.id} after {past} steps") from None
            if isinstance(script, str):
                return self._text(script)
            return self._text(_format_plan(script))
        if stage == "executor":
            step = _required(_STEP, blob, "STEP")
            return self._tool_for_step(step)
        if stage == "react":
            return self._react(task, messages)
        if stage == "rewoo_planner":
            lines = [f"Plan: solve {task.id}.", *task.rewoo_lines]
            return self._text("\n".join(lines))
        if stage == "rewoo_solver":
            return self._text(task.answer)
        raise ValueError(f"unknown control stage {stage!r}")

    def _text(self, content: str) -> LLMResponse:
        return LLMResponse(content=content, model=self.name)

    def _tool_for_step(self, step: str) -> LLMResponse:
        if ":" not in step:
            return self._text(f"ERROR: cannot execute {step}")
        name, argument = step.split(":", 1)
        name = name.strip()
        argument = argument.strip()
        if name == "search":
            call = ToolCall(name="search", arguments={"query": argument}, call_id="exec")
        elif name == "calculator":
            call = ToolCall(name="calculator", arguments={"expression": argument}, call_id="exec")
        else:
            return self._text(f"ERROR: unknown tool in step {step}")
        return LLMResponse(content="", tool_calls=[call], model=self.name)

    def _react(self, task: ControlTask, messages: list) -> LLMResponse:
        seen = sum(1 for message in messages if message.role == MessageRole.TOOL)
        if seen >= len(task.react_steps) or task.react_steps[seen] is None:
            return self._text(task.answer)
        step = task.react_steps[seen]
        if step is None:
            return self._text(task.answer)
        name, arguments = step
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(name=name, arguments=dict(arguments), call_id=f"react-{seen + 1}")],
            model=self.name,
        )


def _required(pattern: re.Pattern[str], blob: str, label: str) -> str:
    found = pattern.search(blob)
    if found is None:
        raise ValueError(f"control prompt missing {label}")
    return found.group(1).strip()


def _format_plan(steps: tuple[str, ...]) -> str:
    lines = ["PLAN:"]
    lines.extend(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    return "\n".join(lines)
