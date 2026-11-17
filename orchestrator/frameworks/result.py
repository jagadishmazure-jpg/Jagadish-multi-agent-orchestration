"""Shared result type for one research-and-report run."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.meter import MeteredLLM
from .golden import case_by_id, match_case, quality_score


@dataclass
class ResearchRun:
    """Output of one framework on one topic, plus the meter totals for that run."""

    framework: str
    runtime: str
    topic: str
    case_id: str
    final: str
    stages: dict[str, str] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def quality(self) -> float:
        case = case_by_id(self.case_id) if self.case_id else match_case(self.topic)
        return quality_score(self.final, case)


def check_topic(topic: str) -> str:
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string")
    return topic.strip()


def finish(
    framework: str,
    runtime: str,
    topic: str,
    stages: dict[str, str],
    meter: MeteredLLM,
) -> ResearchRun:
    case = match_case(topic)
    return ResearchRun(
        framework=framework,
        runtime=runtime,
        topic=topic,
        case_id=case["id"] if case else "",
        final=stages.get("review", ""),
        stages=stages,
        prompt_tokens=meter.prompt_tokens,
        completion_tokens=meter.completion_tokens,
        latency_ms=meter.latency_ms,
        llm_calls=meter.calls,
    )
