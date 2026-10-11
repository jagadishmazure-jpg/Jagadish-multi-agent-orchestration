"""Agent base class: name, role, and an LLM backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..llm.client import LLMClient
from .messages import Message, MessageRole

if TYPE_CHECKING:  # avoid a hard tools -> core import cycle
    from ..tools.registry import Tool


class Agent:
    """A named agent with a role, backed by an LLM client.

    The default LLM is a deterministic fake, so agents run fully offline.
    """

    def __init__(
        self,
        name: str,
        role: str,
        llm: LLMClient,
        tools: list[Tool] | None = None,
        system_prompt: str = "",
    ) -> None:
        if not name:
            raise ValueError("agent name must be non-empty")
        self.name = name
        self.role = role
        self.llm = llm
        self.tools: list[Tool] = list(tools or [])
        self.system_prompt = system_prompt or f"You are {name}, a {role}."

    def tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def build_messages(self, conversation: list[Message]) -> list[Message]:
        return [Message(MessageRole.SYSTEM, self.system_prompt)] + list(conversation)

    def respond(self, conversation: list[Message]) -> Message:
        """Produce one assistant message for the given conversation."""
        response = self.llm.complete(
            self.build_messages(conversation),
            tools=self.tool_schemas() or None,
        )
        return Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
            sender=self.name,
            metadata={
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments, "call_id": tc.call_id}
                    for tc in response.tool_calls
                ],
                "model": response.model,
            },
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Agent(name={self.name!r}, role={self.role!r})"
