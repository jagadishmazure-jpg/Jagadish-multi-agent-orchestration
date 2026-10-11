"""LLM client interface and deterministic fake implementations."""

from .client import LLMClient, LLMResponse, ToolCall
from .fake import FakeLLM

__all__ = ["FakeLLM", "LLMClient", "LLMResponse", "ToolCall"]
