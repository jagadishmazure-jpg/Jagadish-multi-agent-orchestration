"""Research-and-report as a CrewAI crew.

Four agents (planner, researcher, writer, critic) and four tasks run with
``Process.sequential``. Each task sees the previous task's output, which is
the crew equivalent of the existing research pipeline.

``backend="builtin"`` runs that process on :class:`ResearchMockLLM` with the
crew's role, goal, backstory, and task card in the prompt. ``backend="crewai"``
builds a real ``crewai.Crew`` from the same specs and kicks it off. The
scripted model still answers from ``ROLE:``, so the article matches the golden
set offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from ..core.messages import Message, MessageRole
from ..llm.factory import azure_config, build_llm
from ..llm.meter import MeteredLLM
from .mock import ResearchMockLLM
from .result import ResearchRun, check_topic, finish

_ROLE_BY_TITLE = {
    "Research Planner": "planner",
    "Research Analyst": "researcher",
    "Technical Writer": "writer",
    "Critic": "critic",
}


@dataclass(frozen=True)
class CrewMember:
    """One crew agent. ``key`` is the golden-set stage the mock routes on."""

    key: str
    role: str
    goal: str
    backstory: str


@dataclass(frozen=True)
class CrewTaskSpec:
    """One sequential task. ``description`` may contain ``{topic}``."""

    key: str
    description: str
    expected_output: str
    agent: str


MEMBERS: tuple[CrewMember, ...] = (
    CrewMember(
        "planner",
        "Research Planner",
        "Break the topic into a short research plan.",
        "You plan the work before anyone gathers facts.",
    ),
    CrewMember(
        "researcher",
        "Research Analyst",
        "Collect the names, dates, and the core idea for the topic.",
        "You only report a fact you can attach to a person, a place, or a year.",
    ),
    CrewMember(
        "writer",
        "Technical Writer",
        "Turn the findings into a short article.",
        "You write plain prose a reviewer can check against the findings.",
    ),
    CrewMember(
        "critic",
        "Critic",
        "Approve the article only when the names and years are present.",
        "You reject a draft that omits who was involved or when it happened.",
    ),
)

TASKS: tuple[CrewTaskSpec, ...] = (
    CrewTaskSpec(
        "plan",
        "Write a research plan for {topic}.",
        "A numbered RESEARCH PLAN.",
        "planner",
    ),
    CrewTaskSpec(
        "findings",
        "Research {topic} using the plan from the previous task.",
        "A FINDINGS brief.",
        "researcher",
    ),
    CrewTaskSpec(
        "draft",
        "Write a draft article about {topic} from the findings.",
        "A DRAFT ARTICLE.",
        "writer",
    ),
    CrewTaskSpec(
        "review",
        "Review the draft article about {topic}.",
        "A CRITIC REVIEW that says APPROVED or REJECTED.",
        "critic",
    ),
)

_STAGE = {"plan": "plan", "findings": "findings", "draft": "draft", "review": "review"}


def run_crewai_research(
    topic: str,
    llm: Any = None,
    *,
    backend: str = "builtin",
) -> ResearchRun:
    """Run the research crew. ``backend`` is ``builtin``, ``crewai``, or ``auto``."""
    topic = check_topic(topic)
    chosen = _resolve_backend(backend)
    inner = llm if llm is not None else _default_llm()
    meter = MeteredLLM(inner)
    if chosen == "crewai":
        stages = _run_library(topic, meter)
        runtime = "crewai"
    else:
        stages = _run_builtin(topic, meter)
        runtime = "builtin"
    return finish("crewai", runtime, topic, stages, meter)


def _resolve_backend(backend: str) -> str:
    if backend not in {"auto", "builtin", "crewai"}:
        raise ValueError("backend must be 'auto', 'builtin', or 'crewai'")
    if backend == "auto":
        return "crewai" if find_spec("crewai") is not None else "builtin"
    if backend == "crewai" and find_spec("crewai") is None:
        raise ImportError(
            "crewai is not installed. Run: python -m pip install "
            "'jagadish-multi-agent-orchestration[frameworks]'"
        )
    return backend


def _default_llm() -> Any:
    if azure_config() is not None:
        return build_llm()
    return ResearchMockLLM()


def _run_builtin(topic: str, meter: MeteredLLM) -> dict[str, str]:
    members = {member.key: member for member in MEMBERS}
    stages: dict[str, str] = {}
    previous = ""
    for spec in TASKS:
        member = members[spec.agent]
        response = meter.complete(_crew_messages(member, spec, topic, previous))
        stages[_STAGE[spec.key]] = response.content
        previous = response.content
    return stages


def _crew_messages(
    member: CrewMember, spec: CrewTaskSpec, topic: str, previous: str
) -> list[Message]:
    system = (
        f"ROLE: {member.key}\n"
        "You are a CrewAI agent in a sequential crew.\n"
        f"Role: {member.role}\n"
        f"Goal: {member.goal}\n"
        f"Backstory: {member.backstory}\n"
    )
    lines = [
        f"Task: {spec.description.format(topic=topic)}",
        f"Expected output: {spec.expected_output}",
        f"Topic: {topic}",
    ]
    if previous:
        lines.append("Context from the previous task:")
        lines.append(previous)
    return [
        Message(MessageRole.SYSTEM, system),
        Message(MessageRole.USER, "\n".join(lines)),
    ]


def _run_library(topic: str, meter: MeteredLLM) -> dict[str, str]:
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    from crewai import Agent, Crew, Process, Task
    from crewai.llms.base_llm import BaseLLM

    class ScriptedCrewLLM(BaseLLM):
        """CrewAI LLM that delegates to the shared meter and mock."""

        def __init__(self) -> None:
            super().__init__(model="research-mock", temperature=0.0)

        def call(
            self,
            messages: str | list,
            tools: list | None = None,
            callbacks: list | None = None,
            available_functions: dict | None = None,
            from_task: Any = None,
            from_agent: Any = None,
            response_model: Any = None,
        ) -> str:
            title = getattr(from_agent, "role", "") if from_agent is not None else ""
            key = _ROLE_BY_TITLE.get(str(title), "")
            routed = [
                Message(MessageRole.SYSTEM, f"ROLE: {key}\nCrewAI agent {title}".strip()),
                Message(MessageRole.USER, _flatten(messages)),
            ]
            return meter.complete(routed).content

        def supports_function_calling(self) -> bool:
            return False

        def get_input_tokens_count(
            self,
            messages: str | list,
            tools: list | None = None,
            response_model: Any = None,
        ) -> int:
            return max(1, len(_flatten(messages)) // 4)

    scripted = ScriptedCrewLLM()
    agents = {
        member.key: Agent(
            role=member.role,
            goal=member.goal,
            backstory=member.backstory,
            llm=scripted,
            verbose=False,
            allow_delegation=False,
        )
        for member in MEMBERS
    }
    task_objects = []
    for spec in TASKS:
        task_objects.append(
            Task(
                description=spec.description,
                expected_output=spec.expected_output,
                agent=agents[spec.agent],
            )
        )
    for index, task_object in enumerate(task_objects):
        if index:
            task_object.context = [task_objects[index - 1]]
    crew = Crew(
        agents=[agents[member.key] for member in MEMBERS],
        tasks=task_objects,
        process=Process.sequential,
        verbose=False,
        memory=False,
    )
    result = crew.kickoff(inputs={"topic": topic})
    return _stages_from_kickoff(result)


def _stages_from_kickoff(result: Any) -> dict[str, str]:
    outputs = getattr(result, "tasks_output", None) or []
    keys = ["plan", "findings", "draft", "review"]
    stages: dict[str, str] = {}
    for key, item in zip(keys, outputs, strict=False):
        stages[key] = str(getattr(item, "raw", item))
    if "review" not in stages:
        raw = getattr(result, "raw", None)
        stages["review"] = str(raw if raw else result)
    return stages


def _flatten(messages: str | list) -> str:
    if isinstance(messages, str):
        return messages
    parts: list[str] = []
    for message in messages:
        if isinstance(message, str):
            parts.append(message)
        elif isinstance(message, dict):
            parts.append(str(message.get("content", "")))
        else:
            parts.append(str(getattr(message, "content", message)))
    return "\n".join(parts)
