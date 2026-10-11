"""Failure-mode tests: cutoffs, unknown tools, agent errors, bad inputs."""

import pytest

from orchestrator.core import Agent, Message, MessageRole, TaskState
from orchestrator.llm import FakeLLM
from orchestrator.patterns import Blackboard, Router, SequentialPipeline, Supervisor
from orchestrator.tools import (
    MaxIterationsError,
    ToolCallingLoop,
    ToolNotFoundError,
    ToolRegistry,
    make_calculator_tool,
)


class BrokenLLM:
    """LLM backend that always fails."""

    def complete(self, messages, tools=None):
        raise RuntimeError("llm exploded")


def _agent(name: str, llm=None) -> Agent:
    return Agent(name=name, role="test", llm=llm or FakeLLM(default="ok"))


def _tool_loop_agent() -> Agent:
    return Agent(
        name="looper",
        role="test",
        llm=FakeLLM(
            sequence=[
                {
                    "content": "again",
                    "tool_calls": [
                        {"name": "calculator", "arguments": {"expression": "1+1"}}
                    ],
                }
            ]
        ),
        tools=[make_calculator_tool()],
    )


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(make_calculator_tool())
    return reg


def test_max_iteration_cutoff():
    loop = ToolCallingLoop(_tool_loop_agent(), _registry(), max_iterations=3)
    with pytest.raises(MaxIterationsError, match="looper"):
        loop.run("never finishes")


def test_unknown_tool_in_loop():
    agent = Agent(
        name="x",
        role="test",
        llm=FakeLLM(
            sequence=[
                {"content": "?", "tool_calls": [{"name": "nope", "arguments": {}}]},
                "done",
            ]
        ),
    )
    with pytest.raises(ToolNotFoundError):
        ToolCallingLoop(agent, _registry()).run("call unknown")


def test_agent_llm_error_propagates_in_loop():
    agent = _agent("broken", llm=BrokenLLM())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="llm exploded"):
        ToolCallingLoop(agent, _registry()).run("hi")


def test_agent_llm_error_propagates_in_respond():
    agent = _agent("broken", llm=BrokenLLM())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="llm exploded"):
        agent.respond([Message(MessageRole.USER, "hi")])


def test_empty_task_raises_in_pipeline():
    pipe = SequentialPipeline([_agent("a")])
    with pytest.raises(ValueError):
        pipe.run("")


def test_router_unmatched_without_default():
    clf = Agent(
        name="clf",
        role="test",
        llm=FakeLLM(default="CATEGORY: unknown_zone"),
    )
    router = Router(clf, {"a": _agent("a")})
    with pytest.raises(Exception):
        router.run("task")


def test_supervisor_worker_error_propagates():
    decomposer = _agent("d", FakeLLM(default="w1: do it"))
    workers = {"w1": _agent("w1", llm=BrokenLLM())}  # type: ignore[arg-type]
    sup = Supervisor(decomposer, workers, _agent("s"))
    with pytest.raises(RuntimeError, match="llm exploded"):
        sup.run("task")


def test_blackboard_never_converges():
    bb = Blackboard([_agent("a", FakeLLM(default="still thinking"))], max_rounds=2)
    result = bb.run("task")
    assert result.converged is False
    assert result.rounds == 2
    assert "a" in result.board  # contribution recorded despite no DONE


def test_tool_missing_argument_in_loop():
    agent = Agent(
        name="x",
        role="test",
        llm=FakeLLM(
            sequence=[
                {
                    "content": "?",
                    "tool_calls": [{"name": "calculator", "arguments": {}}],
                },
                "done",
            ]
        ),
        tools=[make_calculator_tool()],
    )
    from orchestrator.tools import ToolError

    with pytest.raises(ToolError):
        ToolCallingLoop(agent, _registry()).run("bad args")


def test_state_failed_after_max_iterations():
    state = TaskState(task="t")
    loop = ToolCallingLoop(_tool_loop_agent(), _registry(), max_iterations=1)
    with pytest.raises(MaxIterationsError):
        loop.run("t", state=state)
    assert not state.ok
    assert state.errors
