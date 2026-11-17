"""Plan-and-execute and ReWOO against the existing ReAct tool loop."""

import importlib.util

import pytest

from orchestrator.llm import MOCK_CALL_MS
from orchestrator.llm.client import LLMResponse, ToolCall
from orchestrator.patterns import TASKS, compare_control, run_plan_execute, run_react, run_rewoo
from orchestrator.patterns.control_llm import ControlMockLLM
from orchestrator.patterns.tasks import task_by_id


def test_all_three_patterns_solve_every_task():
    scores = compare_control(llm=ControlMockLLM(), backend="builtin")
    by_name = {score.pattern: score for score in scores}
    assert [score.pattern for score in scores] == ["react", "plan_execute", "rewoo"]
    for score in scores:
        assert score.success_rate == 1.0
        assert score.passed == score.tasks == len(TASKS)
        assert score.latency_ms == score.llm_calls * MOCK_CALL_MS
    assert by_name["rewoo"].llm_calls < by_name["react"].llm_calls < by_name["plan_execute"].llm_calls
    assert by_name["rewoo"].total_tokens < by_name["react"].total_tokens
    assert by_name["rewoo"].latency_ms < by_name["plan_execute"].latency_ms
    again = compare_control(llm=ControlMockLLM(), backend="builtin")
    assert [(s.total_tokens, s.llm_calls, s.success_rate) for s in scores] == [
        (s.total_tokens, s.llm_calls, s.success_rate) for s in again
    ]


def test_rewoo_uses_two_model_calls_and_threads_evidence():
    task = task_by_id("capitals_population_sum")
    result = run_rewoo(task, llm=ControlMockLLM(), backend="builtin")
    assert result.success
    assert result.llm_calls == 2
    assert result.visited == ["planner", "worker", "solver"]
    assert result.evidence["E1"] == "Paris"
    assert result.evidence["E2"] == "Tokyo"
    assert result.evidence["E5"] == "16100000"
    assert result.tool_trace[-1]["arguments"]["expression"] == "2100000+14000000"


def test_plan_execute_revises_an_incomplete_plan():
    capitals = run_plan_execute(task_by_id("capitals_population_sum"), llm=ControlMockLLM())
    assert capitals.success
    assert capitals.revisions == 2
    assert capitals.visited[0] == "planner"
    assert capitals.visited[-1] == "replanner"
    assert "executor" in capitals.visited
    assert capitals.answer == "16100000"
    canberra = run_plan_execute(task_by_id("canberra_twice"), llm=ControlMockLLM())
    assert canberra.revisions == 2
    assert canberra.answer == "940000"
    identity = run_plan_execute(task_by_id("python_identity"), llm=ControlMockLLM())
    assert identity.revisions == 0
    assert identity.answer == "Guido van Rossum, 1991"


def test_react_records_tool_results_on_the_existing_loop():
    result = run_react(task_by_id("python_identity"), llm=ControlMockLLM())
    assert result.success
    assert result.runtime == "tool_loop"
    assert result.trajectory is not None
    assert [event.detail["tool"] for event in result.trajectory.filter("tool_result")] == [
        "search",
        "search",
    ]
    assert "Guido van Rossum" in result.tool_trace[0]["result"]


def test_react_honours_the_iteration_budget():
    class _AlwaysTool:
        def complete(self, messages, tools=None):
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(name="search", arguments={"query": "capital of France"})],
                model="loop",
            )

    result = run_react(task_by_id("canberra_twice"), llm=_AlwaysTool(), max_iterations=2)
    assert result.success is False
    assert result.llm_calls == 2
    assert "2 iterations" in result.error


def test_graph_stops_when_the_replanner_never_finishes():
    class _NeverDone(ControlMockLLM):
        def complete(self, messages, tools=None):
            blob = "\n".join(message.content for message in messages)
            if "STAGE: replanner" in blob:
                return LLMResponse(content="PLAN:\n1. search: capital of France", model="loop")
            return super().complete(messages, tools)

    with pytest.raises(RuntimeError, match="exceeded 4 steps"):
        run_plan_execute(
            task_by_id("python_identity"),
            llm=_NeverDone(),
            backend="builtin",
            max_steps=4,
        )


def test_replanner_mock_requires_the_observation_text():
    from orchestrator.core import Message, MessageRole

    blind = ControlMockLLM().complete(
        [
            Message(
                MessageRole.SYSTEM,
                "STAGE: replanner\nTASK: capitals_population_sum\nPAST: 1",
            ),
            Message(MessageRole.USER, "Completed steps:\n1. search: capital of France -> unknown"),
        ]
    )
    assert blind.content == "FINAL: missing observation"
    seen = ControlMockLLM().complete(
        [
            Message(
                MessageRole.SYSTEM,
                "STAGE: replanner\nTASK: capitals_population_sum\nPAST: 1",
            ),
            Message(MessageRole.USER, "Completed steps:\n1. search: capital of France -> Paris"),
        ]
    )
    assert "capital of Japan" in seen.content


def test_rewoo_rejects_duplicate_and_forward_references():
    from orchestrator.patterns.rewoo import parse_rewoo

    with pytest.raises(ValueError, match="duplicate"):
        parse_rewoo("#E1 = search[capital of France]\n#E1 = search[capital of Japan]")
    with pytest.raises(ValueError, match="before it is produced"):
        parse_rewoo("#E2 = search[population of #E1]")


def test_rewoo_max_steps_bounds_tool_calls():
    from orchestrator.llm.client import LLMResponse

    class _LongPlan(ControlMockLLM):
        def complete(self, messages, tools=None):
            blob = "\n".join(message.content for message in messages)
            if "STAGE: rewoo_planner" in blob:
                lines = "\n".join(f"#E{i} = search[capital of France]" for i in range(1, 5))
                return LLMResponse(content=lines, model="loop")
            return super().complete(messages, tools)

    with pytest.raises(RuntimeError, match="max_steps"):
        run_rewoo(task_by_id("python_identity"), llm=_LongPlan(), backend="builtin", max_steps=2)


def test_rewoo_rejects_a_plan_with_no_steps():
    class _EmptyPlan(ControlMockLLM):
        def complete(self, messages, tools=None):
            blob = "\n".join(message.content for message in messages)
            if "STAGE: rewoo_planner" in blob:
                return LLMResponse(content="Plan: nothing to do.", model="loop")
            return super().complete(messages, tools)

    with pytest.raises(ValueError, match="no #E steps"):
        run_rewoo(task_by_id("python_identity"), llm=_EmptyPlan(), backend="builtin")


def test_langgraph_backend_matches_the_builtin_when_installed():
    if importlib.util.find_spec("langgraph") is None:
        pytest.skip("langgraph is not installed")
    task = task_by_id("capitals_population_sum")
    builtin = run_plan_execute(task, llm=ControlMockLLM(), backend="builtin")
    live = run_plan_execute(task, llm=ControlMockLLM(), backend="langgraph")
    assert live.runtime == "langgraph"
    assert live.answer == builtin.answer
    assert live.revisions == builtin.revisions
    assert live.prompt_tokens == builtin.prompt_tokens
    assert live.visited == builtin.visited
    rewoo_live = run_rewoo(task, llm=ControlMockLLM(), backend="langgraph")
    rewoo_builtin = run_rewoo(task, llm=ControlMockLLM(), backend="builtin")
    assert rewoo_live.success
    assert rewoo_live.evidence == rewoo_builtin.evidence
    assert rewoo_live.llm_calls == 2
