"""Wrap an LLM client and record tokens and latency for a comparison run."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from .tokens import count_text

if TYPE_CHECKING:
    from .client import LLMClient, LLMResponse

# Mock latency is a fixed cost per completion so the published tables stay
# deterministic. Live Azure calls replace it with measured wall-clock time.
MOCK_CALL_MS = 5


def count_messages(messages: list) -> int:
    """Estimate prompt tokens for a list of orchestrator messages."""
    total = 0
    for message in messages:
        role = message.role.value if hasattr(message.role, "value") else str(message.role)
        total += count_text(role)
        total += count_text(message.content)
    return total


class MeteredLLM:
    """Delegate completions and accumulate token and latency totals.

    When the inner client reports ``usage["source"] == "azure"``, token totals
    come from the service and latency is wall-clock milliseconds. Otherwise
    tokens are estimated from the text and each call costs ``call_ms``.
    """

    def __init__(self, inner: LLMClient, call_ms: int = MOCK_CALL_MS) -> None:
        if call_ms < 0:
            raise ValueError("call_ms must be >= 0")
        self.inner = inner
        self.call_ms = call_ms
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.latency_ms = 0
        self.live = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def complete(self, messages: list, tools: list[dict] | None = None) -> LLMResponse:
        started = time.perf_counter()
        response = self.inner.complete(messages, tools)
        elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))
        usage = dict(response.usage or {})
        if usage.get("source") == "azure":
            prompt = int(usage.get("prompt_tokens") or 0)
            completion = int(usage.get("completion_tokens") or 0)
            self.live = True
            self.latency_ms += elapsed_ms
        else:
            prompt = count_messages(messages)
            completion = count_text(response.content)
            for call in response.tool_calls:
                completion += count_text(call.name)
                completion += count_text(json.dumps(call.arguments, sort_keys=True, default=str))
            self.latency_ms += self.call_ms
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1
        response.usage = {
            **usage,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
        return response
