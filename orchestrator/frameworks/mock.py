"""Scripted research model. Routes on ``ROLE:`` and the topic text.

The same object backs the offline CrewAI, LangGraph, and Agent Framework
runners, so quality differences would come from the orchestration, not from a
different model. On this golden set the scripted answer is identical; the
harness still scores it.
"""

from __future__ import annotations

import re

from ..llm.client import LLMResponse
from .golden import match_case

_ROLE = re.compile(r"^ROLE:\s*(planner|researcher|writer|critic)\s*$", re.MULTILINE)
_STAGE = {
    "planner": "plan",
    "researcher": "findings",
    "writer": "draft",
    "critic": "review",
}
# Later stages must see the previous stage's text. Otherwise the scripted
# model refuses, so a runner that drops the handoff fails the golden set.
_NEEDS_PRIOR = {
    "researcher": "RESEARCH PLAN",
    "writer": "FINDINGS:",
    "critic": "DRAFT ARTICLE",
}
REJECTED = "CRITIC REVIEW: REJECTED. No scripted facts for this topic."


class ResearchMockLLM:
    """Deterministic stand-in used unless Azure OpenAI is requested explicitly."""

    def __init__(self, name: str = "research-mock") -> None:
        self.name = name

    def complete(self, messages: list, tools: list[dict] | None = None) -> LLMResponse:
        blob = "\n".join(message.content for message in messages)
        found = _ROLE.search(blob)
        case = match_case(blob)
        role = found.group(1) if found else ""
        needed = _NEEDS_PRIOR.get(role)
        if found is None or case is None or (needed and needed not in blob):
            content = REJECTED
        else:
            content = str(case[_STAGE[role]])
        return LLMResponse(content=content, model=self.name)
