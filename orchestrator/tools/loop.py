"""Tool-calling agent loop (ReAct-style, fully offline)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.messages import Message, MessageRole
from ..core.state import TaskState
from .registry import ToolRegistry

if TYPE_CHECKING:
    from ..core.agent import Agent
    from ..eval.recorder import TrajectoryRecorder


class MaxIterationsError(Exception):
    """Raised when the agent loop exceeds its iteration budget."""


class ToolCallingLoop:
    """Run an agent until it answers or the iteration budget is exhausted.

    Each iteration: the agent responds; any requested tool calls are executed
    and fed back as ``tool`` messages. Unknown tools and agent errors
    propagate to the caller so failure modes stay visible.
    """

    def __init__(
        self,
        agent: Agent,
        registry: ToolRegistry,
        max_iterations: int = 5,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.agent = agent
        self.registry = registry
        self.max_iterations = max_iterations
        self.recorder = recorder

    def _record(self, kind: str, **detail) -> None:
        if self.recorder is not None:
            self.recorder.record(kind, agent=self.agent.name, **detail)

    def run(self, task: str, state: TaskState | None = None) -> Message:
        state = state if state is not None else TaskState(task=task)
        state.status = "running"
        state.add_message(Message(MessageRole.USER, task, sender="user"))
        self._record("task_start", task=task)

        for iteration in range(1, self.max_iterations + 1):
            reply = self.agent.respond(state.messages)
            state.add_message(reply)
            state.advance()
            calls = reply.metadata.get("tool_calls", [])
            self._record(
                "agent_reply",
                iteration=iteration,
                content=reply.short(),
                tool_calls=[c["name"] for c in calls],
            )
            if not calls:
                state.mark_done()
                self._record("task_done", iterations=iteration)
                return reply
            for call in calls:
                result = self.registry.call(call["name"], call["arguments"])
                tool_msg = Message(
                    role=MessageRole.TOOL,
                    content=str(result),
                    sender=call["name"],
                    metadata={"call_id": call.get("call_id", "")},
                )
                state.add_message(tool_msg)
                self._record(
                    "tool_result",
                    tool=call["name"],
                    result=str(result)[:200],
                )

        state.mark_failed(f"max iterations ({self.max_iterations}) exceeded")
        self._record("max_iterations", budget=self.max_iterations)
        raise MaxIterationsError(
            f"agent {self.agent.name!r} did not finish within "
            f"{self.max_iterations} iterations"
        )
