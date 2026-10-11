"""Tests for the five orchestration patterns."""

import pytest

from orchestrator.core import Agent
from orchestrator.llm import FakeLLM
from orchestrator.patterns import (
    Blackboard,
    Hierarchical,
    Router,
    SequentialPipeline,
    Supervisor,
    UnknownRouteError,
    UnknownTeamError,
    UnknownWorkerError,
    parse_subtasks,
)


def _agent(
    name: str, default: str = "ok", llm=None, **kwargs
) -> Agent:
    backend = llm if llm is not None else FakeLLM(default=default, **kwargs)
    return Agent(name=name, role="test", llm=backend)


# --- supervisor ---


def _supervisor():
    decomposer = _agent(
        "decomposer", default="researcher: find facts\nwriter: write summary"
    )
    workers = {
        "researcher": _agent("researcher", default="facts here"),
        "writer": _agent("writer", default="summary here"),
    }
    synth = _agent("synth", default="FINAL")
    return Supervisor(decomposer, workers, synth)


def test_supervisor_end_to_end():
    result = _supervisor().run("do research")
    assert result.subtasks == [("researcher", "find facts"), ("writer", "write summary")]
    assert result.worker_outputs == {"researcher": "facts here", "writer": "summary here"}
    assert result.final == "FINAL"
    assert len(result.trajectory) > 0


def test_supervisor_unknown_worker_raises():
    sup = _supervisor()
    sup.decomposer = _agent("decomposer", default="ghost: do things")
    with pytest.raises(UnknownWorkerError):
        sup.run("task")


def test_supervisor_empty_plan_raises():
    sup = _supervisor()
    sup.decomposer = _agent("decomposer", default="no actionable lines here")
    with pytest.raises(ValueError):
        sup.run("task")


def test_supervisor_needs_workers():
    with pytest.raises(ValueError):
        Supervisor(_agent("d"), {}, _agent("s"))


def test_parse_subtasks_bullets():
    text = "- alpha: first task\n* beta: second task\nnot a task"
    assert parse_subtasks(text) == [("alpha", "first task"), ("beta", "second task")]


def test_parse_subtasks_nothing_raises():
    with pytest.raises(ValueError):
        parse_subtasks("nothing parseable")


# --- router ---


def _router(**kwargs):
    classifier = _agent(
        "clf",
        llm=FakeLLM(
            rules=[("bill", "CATEGORY: billing"), ("crash", "CATEGORY: technical")],
            default="CATEGORY: general",
        ),
    )
    specialists = {
        "billing": _agent("billing", default="billing answer"),
        "technical": _agent("tech", default="tech answer"),
    }
    return Router(classifier, specialists, **kwargs)


def test_router_routes_billing():
    result = _router().run("my bill is wrong")
    assert result.category == "billing"
    assert result.agent_name == "billing"
    assert result.output == "billing answer"


def test_router_routes_technical():
    result = _router().run("app crash on start")
    assert result.category == "technical"
    assert result.output == "tech answer"


def test_router_unknown_category_raises():
    with pytest.raises(UnknownRouteError):
        _router().run("something else entirely")


def test_router_default_fallback():
    result = _router(default="billing").run("something else")
    assert result.category == "billing"


def test_router_bad_default_raises():
    with pytest.raises(ValueError):
        _router(default="ghost")


def test_router_no_category_in_response_raises():
    clf = _agent("clf", default="I refuse to categorize")
    router = Router(clf, {"a": _agent("a")})
    with pytest.raises(UnknownRouteError):
        router.run("task")


def test_router_needs_specialists():
    with pytest.raises(ValueError):
        Router(_agent("c"), {})


# --- sequential ---


def test_sequential_order_and_final():
    pipe = SequentialPipeline([_agent("a", default="A-out"), _agent("b", default="B-out")])
    result = pipe.run("start")
    assert result.outputs == ["A-out", "B-out"]
    assert result.final == "B-out"
    assert [e.agent for e in result.trajectory.filter("step")] == ["a", "b"]


def test_sequential_single_agent():
    result = SequentialPipeline([_agent("solo", default="done")]).run("t")
    assert result.final == "done"


def test_sequential_empty_agents_raises():
    with pytest.raises(ValueError):
        SequentialPipeline([])


def test_sequential_chains_previous_output():
    seen: list[str] = []

    class Spy(FakeLLM):
        def complete(self, messages, tools=None):
            seen.append(messages[-1].content)
            return super().complete(messages, tools)

    a1 = Agent(name="a1", role="r", llm=Spy(default="OUT-1"))
    a2 = Agent(name="a2", role="r", llm=Spy(default="OUT-2"))
    SequentialPipeline([a1, a2]).run("TASK")
    assert "TASK" in seen[0]
    assert "OUT-1" in seen[1]


# --- blackboard ---


def test_blackboard_converges_when_all_done():
    bb = Blackboard([_agent("a"), _agent("b")], done_marker="ok")
    result = bb.run("task")
    assert result.converged is True
    assert result.rounds == 1
    assert result.board["a_status"] == "done"


def test_blackboard_collects_contributions():
    bb = Blackboard(
        [_agent("a", default="idea one"), _agent("b", default="DONE")],
        done_marker="DONE",
    )
    result = bb.run("task")
    assert result.board["a"] == "idea one"
    assert result.board["b_status"] == "done"


def test_blackboard_max_rounds_stops():
    bb = Blackboard([_agent("a", default="still working")], max_rounds=2)
    result = bb.run("task")
    assert result.converged is False
    assert result.rounds == 2


def test_blackboard_needs_agents():
    with pytest.raises(ValueError):
        Blackboard([])


def test_blackboard_bad_max_rounds():
    with pytest.raises(ValueError):
        Blackboard([_agent("a")], max_rounds=0)


# --- hierarchical ---


def _hierarchical():
    manager = Agent(
        name="manager",
        role="manager",
        llm=FakeLLM(
            sequence=[
                "team_a: research the topic\nteam_b: draft examples",
                "AGGREGATED: combined report",
            ]
        ),
    )
    teams = {
        "team_a": [_agent("a1", default="research done")],
        "team_b": [_agent("b1", default="examples done"), _agent("b2", default="polished")],
    }
    return Hierarchical(manager, teams)


def test_hierarchical_end_to_end():
    result = _hierarchical().run("big task")
    assert result.team_outputs == {"team_a": "research done", "team_b": "polished"}
    assert result.final == "AGGREGATED: combined report"


def test_hierarchical_unknown_team_raises():
    h = _hierarchical()
    h.manager = _agent("manager", default="ghost_team: do it")
    with pytest.raises(UnknownTeamError):
        h.run("task")


def test_hierarchical_needs_teams():
    with pytest.raises(ValueError):
        Hierarchical(_agent("m"), {})
