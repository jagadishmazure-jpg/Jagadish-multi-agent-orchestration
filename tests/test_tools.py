"""Tests for the tool registry, built-in tools, and the tool-calling loop."""

import pytest

from orchestrator.core import Agent, MessageRole, TaskState
from orchestrator.llm import FakeLLM
from orchestrator.tools import (
    MaxIterationsError,
    Tool,
    ToolArgumentError,
    ToolCallingLoop,
    ToolError,
    ToolNotFoundError,
    ToolRegistry,
    make_calculator_tool,
    make_clock_tool,
    make_search_stub_tool,
)


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(make_calculator_tool())
    reg.register(make_search_stub_tool())
    reg.register(make_clock_tool())
    return reg


def test_register_and_get(registry: ToolRegistry):
    assert registry.get("calculator").name == "calculator"
    assert "calculator" in registry
    assert len(registry) == 3


def test_register_duplicate_raises(registry: ToolRegistry):
    with pytest.raises(ValueError):
        registry.register(make_calculator_tool())


def test_get_unknown_raises(registry: ToolRegistry):
    with pytest.raises(ToolNotFoundError):
        registry.get("nope")


def test_call_happy_path(registry: ToolRegistry):
    assert registry.call("calculator", {"expression": "2+3"}) == 5


def test_call_missing_argument_raises(registry: ToolRegistry):
    with pytest.raises(ToolArgumentError):
        registry.call("calculator", {})


def test_call_wrong_type_raises(registry: ToolRegistry):
    with pytest.raises(ToolArgumentError):
        registry.call("calculator", {"expression": 123})


def test_call_unknown_tool_raises(registry: ToolRegistry):
    with pytest.raises(ToolNotFoundError):
        registry.call("ghost", {})


def test_call_wraps_unexpected_exception():
    reg = ToolRegistry()

    def bad() -> None:
        raise RuntimeError("kaboom")

    reg.register(Tool(name="bad", description="bad", func=bad))
    with pytest.raises(ToolError, match="kaboom"):
        reg.call("bad", {})


def test_schemas_format(registry: ToolRegistry):
    schemas = {s["name"]: s for s in registry.schemas()}
    assert schemas["calculator"]["parameters"] == {"expression": "str"}


def test_calculator_arithmetic(registry: ToolRegistry):
    assert registry.call("calculator", {"expression": "2*(3+4)"}) == 14
    assert registry.call("calculator", {"expression": "10/4"}) == 2.5
    assert registry.call("calculator", {"expression": "-3 + 2**3"}) == 5


def test_calculator_division_by_zero(registry: ToolRegistry):
    with pytest.raises(ToolError, match="division by zero"):
        registry.call("calculator", {"expression": "1/0"})


def test_calculator_invalid_expression(registry: ToolRegistry):
    with pytest.raises(ToolError):
        registry.call("calculator", {"expression": "2+"})


def test_calculator_rejects_names(registry: ToolRegistry):
    with pytest.raises(ToolError):
        registry.call("calculator", {"expression": "__import__('os')"})


def test_search_stub_hit(registry: ToolRegistry):
    result = registry.call("search", {"query": "I have a login error"})
    assert "KB-1042" in result


def test_search_stub_miss(registry: ToolRegistry):
    result = registry.call("search", {"query": "quantum toasters"})
    assert "No results" in result


def test_search_stub_case_insensitive():
    reg = ToolRegistry()
    reg.register(make_search_stub_tool({"Hello": "world"}))
    assert reg.call("search", {"query": "HELLO there"}) == "world"


def test_clock_fixed_time(registry: ToolRegistry):
    assert registry.call("clock", {}) == "2026-01-01T00:00:00+00:00"


def test_clock_custom_fixed_time():
    reg = ToolRegistry()
    reg.register(make_clock_tool("2000-01-01T00:00:00+00:00"))
    assert reg.call("clock", {}) == "2000-01-01T00:00:00+00:00"


def _agent_with(sequence) -> Agent:
    return Agent(
        name="toolbot",
        role="tool user",
        llm=FakeLLM(sequence=sequence),
        tools=[make_calculator_tool()],
    )


def test_loop_happy_path_no_tools(registry: ToolRegistry):
    agent = Agent(name="plain", role="r", llm=FakeLLM(default="done"))
    loop = ToolCallingLoop(agent, registry)
    reply = loop.run("say hi")
    assert reply.content == "done"


def test_loop_executes_tool_calls(registry: ToolRegistry):
    agent = _agent_with(
        [
            {
                "content": "computing",
                "tool_calls": [
                    {"name": "calculator", "arguments": {"expression": "6*7"}}
                ],
            },
            "The answer is 42.",
        ]
    )
    loop = ToolCallingLoop(agent, registry)
    reply = loop.run("what is 6*7?")
    assert "42" in reply.content
    assert agent.llm.calls == 2


def test_loop_tool_result_fed_back(registry: ToolRegistry):
    agent = _agent_with(
        [
            {
                "content": "",
                "tool_calls": [
                    {"name": "calculator", "arguments": {"expression": "1+1"}}
                ],
            },
            "final",
        ]
    )
    state = TaskState(task="t")
    loop = ToolCallingLoop(agent, registry)
    loop.run("t", state=state)
    tool_msgs = [m for m in state.messages if m.role is MessageRole.TOOL]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "2"


def test_loop_max_iterations_raises(registry: ToolRegistry):
    agent = _agent_with(
        [
            {
                "content": "again",
                "tool_calls": [
                    {"name": "calculator", "arguments": {"expression": "1+1"}}
                ],
            }
        ]
    )
    loop = ToolCallingLoop(agent, registry, max_iterations=2)
    with pytest.raises(MaxIterationsError):
        loop.run("loop forever")


def test_loop_marks_state_failed(registry: ToolRegistry):
    agent = _agent_with(
        [
            {
                "content": "again",
                "tool_calls": [
                    {"name": "calculator", "arguments": {"expression": "1+1"}}
                ],
            }
        ]
    )
    state = TaskState(task="t")
    loop = ToolCallingLoop(agent, registry, max_iterations=1)
    with pytest.raises(MaxIterationsError):
        loop.run("t", state=state)
    assert state.status == "failed"


def test_loop_unknown_tool_raises(registry: ToolRegistry):
    agent = _agent_with(
        [
            {
                "content": "x",
                "tool_calls": [{"name": "ghost", "arguments": {}}],
            },
            "never",
        ]
    )
    loop = ToolCallingLoop(agent, registry)
    with pytest.raises(ToolNotFoundError):
        loop.run("use ghost tool")


def test_loop_agent_error_propagates(registry: ToolRegistry):
    class Broken:
        def complete(self, messages, tools=None):
            raise RuntimeError("llm exploded")

    agent = Agent(name="broken", role="r", llm=Broken())  # type: ignore[arg-type]
    loop = ToolCallingLoop(agent, registry)
    with pytest.raises(RuntimeError, match="llm exploded"):
        loop.run("hi")


def test_loop_rejects_zero_iterations(registry: ToolRegistry):
    agent = Agent(name="a", role="r", llm=FakeLLM())
    with pytest.raises(ValueError):
        ToolCallingLoop(agent, registry, max_iterations=0)
