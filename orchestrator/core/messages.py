"""Message types exchanged between agents."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum


class MessageRole(str, Enum):
    """Roles in a multi-agent conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


_seq = itertools.count()


@dataclass
class Message:
    """A single message in a multi-agent conversation.

    ``sender``/``recipient`` name the agents involved; ``metadata`` carries
    structured extras such as tool calls.
    """

    role: MessageRole
    content: str
    sender: str = ""
    recipient: str = ""
    metadata: dict = field(default_factory=dict)
    seq: int = field(default_factory=lambda: next(_seq))

    def __post_init__(self) -> None:
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)
        # Empty content is legal: tool-call-only assistant messages carry
        # their payload in metadata.
        if not isinstance(self.content, str):
            raise ValueError("message content must be a string")

    def short(self, limit: int = 80) -> str:
        """One-line summary useful for logs and trajectories."""
        text = self.content.replace("\n", " ")
        return text if len(text) <= limit else text[: limit - 3] + "..."
