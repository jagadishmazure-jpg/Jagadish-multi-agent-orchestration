"""Research-team scenario: planner -> researcher -> writer -> critic.

A sequential pipeline of four scripted agents that produces a short article.
Fully offline and deterministic; used by the regression eval.
"""

from __future__ import annotations

from orchestrator.core import Agent
from orchestrator.eval import TrajectoryRecorder
from orchestrator.llm import FakeLLM
from orchestrator.patterns import SequentialPipeline


def build_research_team(recorder: TrajectoryRecorder | None = None) -> SequentialPipeline:
    planner = Agent(
        name="planner",
        role="research planner",
        llm=FakeLLM(
            default=(
                "RESEARCH PLAN:\n"
                "1. Gather key facts and dates\n"
                "2. Identify the creator and origins\n"
                "3. Draft the findings"
            )
        ),
    )
    researcher = Agent(
        name="researcher",
        role="researcher",
        llm=FakeLLM(
            rules=[
                (
                    "research plan",
                    "FINDINGS: Python was created by Guido van Rossum and first "
                    "released in 1991. Its design emphasizes readability with "
                    "significant indentation. Python 3 arrived in 2008.",
                )
            ],
            default="FINDINGS: no plan matched; reporting generic findings.",
        ),
    )
    writer = Agent(
        name="writer",
        role="technical writer",
        llm=FakeLLM(
            rules=[
                (
                    "findings",
                    "DRAFT ARTICLE: Python, created by Guido van Rossum and "
                    "released in 1991, is a readable, indentation-based "
                    "language that grew into one of the world's most popular "
                    "programming languages.",
                )
            ],
            default="DRAFT ARTICLE: no findings to write up.",
        ),
    )
    critic = Agent(
        name="critic",
        role="reviewer",
        llm=FakeLLM(
            rules=[
                (
                    "draft",
                    "CRITIC REVIEW: APPROVED. The Python article correctly "
                    "names the creator (Guido van Rossum) and the release "
                    "year (1991).",
                )
            ],
            default="CRITIC REVIEW: REJECTED. No draft was provided.",
        ),
    )
    return SequentialPipeline(
        [planner, researcher, writer, critic],
        recorder=recorder if recorder is not None else TrajectoryRecorder(),
    )


def run_research_scenario(topic: str = "The history of Python") -> dict:
    """Run the research team end-to-end and return a result bundle."""
    recorder = TrajectoryRecorder()
    pipeline = build_research_team(recorder=recorder)
    result = pipeline.run(f"Research: {topic}")
    agents = [a.name for a in pipeline.agents]
    return {
        "topic": topic,
        "final": result.final,
        "outputs": result.outputs,
        "agents": agents,
        "trajectory": recorder,
        "errors": [],
    }
