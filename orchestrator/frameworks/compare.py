"""Compare CrewAI, LangGraph, and Microsoft Agent Framework on the golden set."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .crewai_research import run_crewai_research
from .golden import load_cases
from .langgraph_research import run_langgraph_research
from .maf_research import run_maf_research
from .result import ResearchRun

_MODULES = {
    "crewai": "orchestrator.frameworks.crewai_research",
    "langgraph": "orchestrator.frameworks.langgraph_research",
    "agent_framework": "orchestrator.frameworks.maf_research",
}

_LIBRARY_BACKEND = {
    "crewai": "crewai",
    "langgraph": "langgraph",
    "agent_framework": "agent_framework",
}


@dataclass
class FrameworkScore:
    """Totals for one framework across the golden research cases."""

    framework: str
    quality: float
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    llm_calls: int
    code_lines: int
    passed: int
    cases: int
    runtime: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def code_lines_for(framework: str) -> int:
    """Non-blank, non-comment lines in that framework's module.

    Shared pieces (the golden set, the meter, the mock) are not included, so
    the number is the size of the framework-specific runner.
    """
    module = importlib.import_module(_MODULES[framework])
    file = getattr(module, "__file__", None)
    if not file:
        raise ValueError(f"no source file for {framework}")
    return count_code_lines(Path(file))


def count_code_lines(path: Path) -> int:
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        total += 1
    return total


def compare_frameworks(
    llm: Any = None,
    *,
    backend: str = "builtin",
) -> list[FrameworkScore]:
    """Score the three research runners on every golden case.

    ``backend="builtin"`` is the deterministic offline engine. ``backend="library"``
    builds a real Crew, StateGraph, and SequentialBuilder. Pass an explicit
    ``llm`` to force the mock; omit it and the runners use Azure OpenAI when
    that environment is configured.
    """
    if backend not in {"builtin", "library"}:
        raise ValueError("backend must be 'builtin' or 'library'")
    cases = load_cases()
    runners = (
        ("crewai", run_crewai_research),
        ("langgraph", run_langgraph_research),
        ("agent_framework", run_maf_research),
    )
    scores: list[FrameworkScore] = []
    for name, runner in runners:
        chosen = "builtin" if backend == "builtin" else _LIBRARY_BACKEND[name]
        runs = [runner(case["topic"], llm=llm, backend=chosen) for case in cases]
        scores.append(_score(name, runs))
    return scores


def _score(framework: str, runs: list[ResearchRun]) -> FrameworkScore:
    count = len(runs)
    passed = sum(1 for run in runs if run.quality == 1.0)
    quality = sum(run.quality for run in runs) / count if count else 0.0
    return FrameworkScore(
        framework=framework,
        quality=quality,
        prompt_tokens=sum(run.prompt_tokens for run in runs),
        completion_tokens=sum(run.completion_tokens for run in runs),
        latency_ms=sum(run.latency_ms for run in runs),
        llm_calls=sum(run.llm_calls for run in runs),
        code_lines=code_lines_for(framework),
        passed=passed,
        cases=count,
        runtime=runs[0].runtime if runs else "",
    )
