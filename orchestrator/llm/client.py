"""LLM client interface shared by real and fake backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..core.messages import Message


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    name: str
    arguments: dict = field(default_factory=dict)
    call_id: str = ""


@dataclass
class LLMResponse:
    """Structured response from an LLM completion."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = "unknown"
    usage: dict = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """Interface every LLM backend must satisfy."""

    def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> LLMResponse:
        """Run one completion over the conversation."""
        ...
