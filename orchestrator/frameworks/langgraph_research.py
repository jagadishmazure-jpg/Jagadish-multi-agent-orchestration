"""Research-and-report as a LangGraph StateGraph.

Nodes are planner, researcher, writer, and critic. Each node reads the topic
and only the previous stage, then writes its own field. That is narrower than
the Agent Framework workflow, which forwards the whole conversation.

``backend="langgraph"`` compiles these same nodes with ``langgraph.graph.StateGraph``.
"""

from __future__ import annotations

from typing import Any, TypedDict

from ..core.messages import Message, MessageRole
from ..llm.factory import azure_config, build_llm
from ..llm.meter import MeteredLLM
from ..patterns.graph import graph_types
from .mock import ResearchMockLLM
from .result import ResearchRun, check_topic, finish


class ResearchState(TypedDict, total=False):
    """Fields the research graph passes between nodes."""

    topic: str
    plan: str
    findings: str
    draft: str
    review: str
    visited: list[str]


def run_langgraph_research(
    topic: str,
    llm: Any = None,
    *,
    backend: str = "builtin",
) -> ResearchRun:
    """Run the four-node research graph."""
    topic = check_topic(topic)
    inner = llm if llm is not None else _default_llm()
    meter = MeteredLLM(inner)
    graph_cls, start, end, runtime = graph_types(backend)
    compiled = _compile(graph_cls, start, end, meter)
    state = compiled.invoke({"topic": topic, "visited": []})
    stages = {
        "plan": str(state.get("plan") or ""),
        "findings": str(state.get("findings") or ""),
        "draft": str(state.get("draft") or ""),
        "review": str(state.get("review") or ""),
    }
    return finish("langgraph", runtime, topic, stages, meter)


def _default_llm() -> Any:
    if azure_config() is not None:
        return build_llm()
    return ResearchMockLLM()


def _compile(graph_cls: Any, start: Any, end: Any, meter: MeteredLLM) -> Any:
    def planner(state: dict) -> dict:
        response = meter.complete(_messages("planner", state["topic"], "", ""))
        return {"plan": response.content}

    def researcher(state: dict) -> dict:
        response = meter.complete(_messages("researcher", state["topic"], "Plan", state.get("plan", "")))
        return {"findings": response.content}

    def writer(state: dict) -> dict:
        response = meter.complete(
            _messages("writer", state["topic"], "Findings", state.get("findings", ""))
        )
        return {"draft": response.content}

    def critic(state: dict) -> dict:
        response = meter.complete(_messages("critic", state["topic"], "Draft", state.get("draft", "")))
        return {"review": response.content}

    graph = graph_cls(ResearchState)
    graph.add_node("planner", _track("planner", planner))
    graph.add_node("researcher", _track("researcher", researcher))
    graph.add_node("writer", _track("writer", writer))
    graph.add_node("critic", _track("critic", critic))
    graph.add_edge(start, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", end)
    return graph.compile()


def _messages(role: str, topic: str, label: str, previous: str) -> list[Message]:
    system = (
        f"ROLE: {role}\n"
        f"You are the {role} node in a LangGraph research graph. "
        "Reply with only your section of the report."
    )
    user = f"Topic: {topic}"
    if previous:
        user = f"{user}\n\n{label}:\n{previous}"
    return [Message(MessageRole.SYSTEM, system), Message(MessageRole.USER, user)]


def _track(name: str, fn: Any) -> Any:
    def inner(state: dict) -> dict:
        update = fn(state) or {}
        seen = list(state.get("visited") or [])
        seen.append(name)
        update["visited"] = seen
        return update

    return inner
