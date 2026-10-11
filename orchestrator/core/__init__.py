"""Core primitives: messages, agents, shared task state, handoff protocol."""

from .agent import Agent
from .handoff import Handoff, perform_handoff
from .messages import Message, MessageRole
from .state import TaskState

__all__ = ["Agent", "Handoff", "Message", "MessageRole", "TaskState", "perform_handoff"]
