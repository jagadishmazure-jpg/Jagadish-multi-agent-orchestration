"""Research-and-report as a Microsoft Agent Framework sequential workflow.

Participants share one conversation. Each agent sees the original request and
every earlier reply, plus its own instructions. That is ``SequentialBuilder``
with the default full-conversation handoff.

``backend="agent_framework"`` builds real ``Agent`` objects and runs
``SequentialBuilder``. The chat client delegates to the same meter the builtin
loop uses, and the instructions strings are the same, so a scripted model
produces the same article.
"""

from __future__ import annotations

import asyncio
from importlib.util import find_spec
from typing import Any

from ..core.messages import Message, MessageRole
from ..llm.factory import azure_config, build_llm
from ..llm.meter import MeteredLLM
from .mock import ResearchMockLLM
from .result import ResearchRun, check_topic, finish

INSTRUCTIONS: dict[str, str] = {
    "planner": (
        "ROLE: planner\n"
        "You are the planner in a sequential agent workflow. Write the research plan."
    ),
    "researcher": (
        "ROLE: researcher\n"
        "You are the researcher in a sequential agent workflow. "
        "Read the conversation and write the findings."
    ),
    "writer": (
        "ROLE: writer\n"
        "You are the writer in a sequential agent workflow. "
        "Read the conversation and draft the article."
    ),
    "critic": (
        "ROLE: critic\n"
        "You are the critic in a sequential agent workflow. "
        "Read the conversation and approve or reject the draft."
    ),
}

_ORDER = ("planner", "researcher", "writer", "critic")
_STAGE = {
    "planner": "plan",
    "researcher": "findings",
    "writer": "draft",
    "critic": "review",
}


def run_maf_research(
    topic: str,
    llm: Any = None,
    *,
    backend: str = "builtin",
) -> ResearchRun:
    """Run the sequential research workflow."""
    topic = check_topic(topic)
    chosen = _resolve_backend(backend)
    inner = llm if llm is not None else _default_llm()
    meter = MeteredLLM(inner)
    if chosen == "agent_framework":
        stages = asyncio.run(_run_library(topic, meter))
        runtime = "agent_framework"
    else:
        stages = _run_builtin(topic, meter)
        runtime = "builtin"
    return finish("agent_framework", runtime, topic, stages, meter)


def _resolve_backend(backend: str) -> str:
    if backend not in {"auto", "builtin", "agent_framework"}:
        raise ValueError("backend must be 'auto', 'builtin', or 'agent_framework'")
    if backend == "auto":
        return "agent_framework" if find_spec("agent_framework") is not None else "builtin"
    if backend == "agent_framework" and find_spec("agent_framework") is None:
        raise ImportError(
            "agent-framework is not installed. Run: python -m pip install "
            "'jagadish-multi-agent-orchestration[frameworks]'"
        )
    return backend


def _default_llm() -> Any:
    if azure_config() is not None:
        return build_llm()
    return ResearchMockLLM()


def _run_builtin(topic: str, meter: MeteredLLM) -> dict[str, str]:
    history = [Message(MessageRole.USER, f"Research: {topic}")]
    stages: dict[str, str] = {}
    for role in _ORDER:
        response = meter.complete(_maf_messages(role, history))
        stages[_STAGE[role]] = response.content
        history.append(Message(MessageRole.ASSISTANT, response.content, sender=role))
    return stages


def _maf_messages(role: str, history: list[Message]) -> list[Message]:
    return [Message(MessageRole.SYSTEM, INSTRUCTIONS[role]), *history]


async def _run_library(topic: str, meter: MeteredLLM) -> dict[str, str]:
    from agent_framework import Agent, BaseChatClient, ChatResponse
    from agent_framework import Message as FrameworkMessage
    from agent_framework.orchestrations import SequentialBuilder

    recorded: dict[str, str] = {}

    class ScriptedChatClient(BaseChatClient):
        """Agent Framework chat client over the shared research meter."""

        async def _inner_get_response(
            self,
            *,
            messages: list,
            stream: bool,
            options: dict,
            **kwargs: Any,
        ) -> ChatResponse:
            if stream:
                raise RuntimeError("the research crew does not stream")
            instructions = str(options.get("instructions") or "")
            converted = [Message(MessageRole.SYSTEM, instructions)]
            for incoming in messages:
                converted.append(Message(_role_of(incoming), _text_of(incoming)))
            response = meter.complete(converted)
            role_name = _role_name(instructions)
            if role_name:
                recorded[_STAGE[role_name]] = response.content
            return ChatResponse(
                messages=[FrameworkMessage(role="assistant", contents=[response.content])]
            )

    client = ScriptedChatClient()
    agents = [
        Agent(client=client, name=role, instructions=INSTRUCTIONS[role]) for role in _ORDER
    ]
    workflow = SequentialBuilder(participants=agents).build()
    result = await workflow.run(f"Research: {topic}")
    outputs = result.get_outputs()
    if outputs and "review" not in recorded:
        recorded["review"] = str(getattr(outputs[-1], "text", outputs[-1]))
    return recorded


def _role_name(instructions: str) -> str:
    for role in _ORDER:
        if f"ROLE: {role}" in instructions:
            return role
    return ""


def _role_of(message: Any) -> MessageRole:
    raw = getattr(message.role, "value", message.role)
    text = str(raw).lower()
    if "assistant" in text:
        return MessageRole.ASSISTANT
    if "system" in text:
        return MessageRole.SYSTEM
    if "tool" in text:
        return MessageRole.TOOL
    return MessageRole.USER


def _text_of(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    return str(text or "")
