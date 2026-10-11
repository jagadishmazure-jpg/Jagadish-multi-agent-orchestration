"""Shared task state passed between agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .messages import Message


@dataclass
class TaskState:
    """Mutable state shared across agents working on one task."""

    task: str
    messages: list[Message] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    status: str = "pending"  # pending | running | done | failed
    errors: list[str] = field(default_factory=list)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def set_artifact(self, key: str, value: Any) -> None:
        self.artifacts[key] = value

    def get_artifact(self, key: str, default: Any = None) -> Any:
        return self.artifacts.get(key, default)

    def advance(self) -> None:
        self.step += 1

    def mark_done(self) -> None:
        self.status = "done"

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.errors.append(error)

    @property
    def ok(self) -> bool:
        return self.status != "failed" and not self.errors
