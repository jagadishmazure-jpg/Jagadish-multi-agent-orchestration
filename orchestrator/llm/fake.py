"""Deterministic fake LLM for fully offline agents and tests.

Responses are scripted in two complementary ways:

* ``rules``: ``(pattern, response)`` pairs. A pattern is either a substring
  (matched case-insensitively against the last message) or a callable
  ``text -> bool``. The first matching rule wins.
* ``sequence``: an ordered list of responses returned one per call; once
  exhausted, the final response repeats. Useful for scripting multi-turn
  tool-calling loops.

A response may be a plain string, an :class:`LLMResponse`, or a dict like
``{"content": "...", "tool_calls": [{"name": ..., "arguments": {...}}]}``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.messages import Message
from .client import LLMResponse, ToolCall

Pattern = str | Callable[[str], bool]
Scripted = str | dict[str, Any] | LLMResponse


class FakeLLM:
    """Scripted, deterministic stand-in for a real LLM client."""

    def __init__(
        self,
        rules: list[tuple[Pattern, Scripted]] | None = None,
        default: Scripted = "Acknowledged.",
        name: str = "fake-llm",
        sequence: list[Scripted] | None = None,
    ) -> None:
        self.rules: list[tuple[Pattern, Scripted]] = list(rules or [])
        self.default = default
        self.name = name
        self.sequence = list(sequence) if sequence else None
        self.calls = 0
        self.history: list[tuple[str, LLMResponse]] = []

    def add_rule(self, pattern: Pattern, response: Scripted) -> None:
        self.rules.append((pattern, response))

    def _match(self, text: str) -> Scripted:
        for pattern, response in self.rules:
            if callable(pattern):
                try:
                    if pattern(text):
                        return response
                except Exception:
                    continue
            elif pattern.lower() in text.lower():
                return response
        return self.default

    def _to_response(self, scripted: Scripted) -> LLMResponse:
        if isinstance(scripted, LLMResponse):
            return scripted
        if isinstance(scripted, dict):
            calls = [
                tc if isinstance(tc, ToolCall) else ToolCall(
                    name=tc["name"],
                    arguments=dict(tc.get("arguments", {})),
                    call_id=tc.get("call_id", ""),
                )
                for tc in scripted.get("tool_calls", [])
            ]
            return LLMResponse(
                content=str(scripted.get("content", "")),
                tool_calls=calls,
                model=self.name,
            )
        return LLMResponse(content=str(scripted), model=self.name)

    def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> LLMResponse:
        self.calls += 1
        last_text = messages[-1].content if messages else ""
        if self.sequence:
            scripted = self.sequence[min(self.calls - 1, len(self.sequence) - 1)]
        else:
            scripted = self._match(last_text)
        response = self._to_response(scripted)
        self.history.append((last_text, response))
        return response
