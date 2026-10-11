"""Router pattern: classify the input, then route to a specialist."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.messages import Message, MessageRole
from ..core.state import TaskState
from ..eval.recorder import TrajectoryRecorder

if TYPE_CHECKING:
    from ..core.agent import Agent

_CATEGORY_RE = re.compile(r"CATEGORY:\s*([A-Za-z0-9_-]+)")


class UnknownRouteError(Exception):
    """Raised when no specialist matches the classified category."""


@dataclass
class RouterResult:
    category: str
    agent_name: str
    output: str
    trajectory: TrajectoryRecorder = field(repr=False)


class Router:
    """Classify with one agent, execute with the matching specialist.

    The classifier must emit ``CATEGORY: <name>`` somewhere in its response.
    """

    def __init__(
        self,
        classifier: Agent,
        specialists: dict[str, Agent],
        default: str | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        if not specialists:
            raise ValueError("router needs at least one specialist")
        if default is not None and default not in specialists:
            raise ValueError(f"default {default!r} is not a known specialist")
        self.classifier = classifier
        self.specialists = specialists
        self.default = default
        self.recorder = recorder if recorder is not None else TrajectoryRecorder()

    def classify(self, task: str) -> str:
        reply = self.classifier.respond([Message(MessageRole.USER, task)])
        match = _CATEGORY_RE.search(reply.content)
        if not match:
            raise UnknownRouteError(
                f"classifier did not emit a category: {reply.content!r}"
            )
        return match.group(1).lower()

    def run(self, task: str) -> RouterResult:
        state = TaskState(task=task, status="running")
        category = self.classify(task)
        self.recorder.record(
            "classify", agent=self.classifier.name, category=category
        )
        if category not in self.specialists:
            if self.default is not None:
                category = self.default
            else:
                raise UnknownRouteError(f"no specialist for {category!r}")
        specialist = self.specialists[category]
        out = specialist.respond([Message(MessageRole.USER, task, sender="router")])
        state.add_message(out)
        state.mark_done()
        self.recorder.record("route", agent=specialist.name, category=category)
        return RouterResult(
            category=category,
            agent_name=specialist.name,
            output=out.content,
            trajectory=self.recorder,
        )
