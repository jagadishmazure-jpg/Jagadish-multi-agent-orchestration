"""Handoff protocol: structured control transfer between agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .messages import Message, MessageRole

if TYPE_CHECKING:  # avoid a hard tools -> core import cycle
    from .agent import Agent
    from .state import TaskState


@dataclass
class Handoff:
    """A structured transfer of control from one agent to another."""

    from_agent: str
    to_agent: str
    payload: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def validate(self) -> None:
        if not self.from_agent:
            raise ValueError("handoff requires a from_agent")
        if not self.to_agent:
            raise ValueError("handoff requires a to_agent")
        if self.from_agent == self.to_agent:
            raise ValueError("handoff target must differ from source")

    def describe(self) -> str:
        self.validate()
        lines = [f"[Handoff from {self.from_agent} to {self.to_agent}"]
        if self.reason:
            lines[0] += f" | reason: {self.reason}"
        lines[0] += "]"
        for key, value in self.payload.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


def perform_handoff(
    handoff: Handoff, agents: dict[str, Agent], state: TaskState
) -> Message:
    """Validate a handoff, record it in state, and build the transfer message.

    Raises:
        ValueError: if the handoff is invalid.
        KeyError: if the target agent is unknown.
    """
    handoff.validate()
    if handoff.to_agent not in agents:
        raise KeyError(f"unknown handoff target: {handoff.to_agent!r}")
    message = Message(
        role=MessageRole.USER,
        content=handoff.describe(),
        sender=handoff.from_agent,
        recipient=handoff.to_agent,
        metadata={"handoff": True, "reason": handoff.reason},
    )
    state.add_message(message)
    state.advance()
    return message
