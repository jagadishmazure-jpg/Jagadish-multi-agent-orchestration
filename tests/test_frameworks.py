"""Golden-set comparison of the research crew across three frameworks."""

import importlib.util
from pathlib import Path

import pytest

from orchestrator.core import Message, MessageRole
from orchestrator.frameworks import compare_frameworks, count_code_lines, load_cases
from orchestrator.frameworks.compare import _MODULES, code_lines_for
from orchestrator.frameworks.crewai_research import run_crewai_research
from orchestrator.frameworks.golden import golden_path, quality_score
from orchestrator.frameworks.langgraph_research import run_langgraph_research
from orchestrator.frameworks.maf_research import run_maf_research
from orchestrator.frameworks.mock import ResearchMockLLM
from orchestrator.llm import MOCK_CALL_MS
from scenarios import run_research_scenario

RUNNERS = {
    "crewai": run_crewai_research,
    "langgraph": run_langgraph_research,
    "agent_framework": run_maf_research,
}


def test_golden_set_has_three_cases_and_matches_the_existing_scenario():
    cases = load_cases()
    assert [case["id"] for case in cases] == [
        "python-history",
        "http-origins",
        "relational-model",
    ]
    python_case = cases[0]
    assert python_case["review"] == run_research_scenario()["final"]
    assert golden_path().name == "golden_research.json"


@pytest.mark.parametrize("framework", list(RUNNERS))
def test_each_framework_passes_the_golden_set(framework):
    runner = RUNNERS[framework]
    for case in load_cases():
        run = runner(case["topic"], llm=ResearchMockLLM(), backend="builtin")
        assert run.framework == framework
        assert run.runtime == "builtin"
        assert run.quality == 1.0
        assert run.final == case["review"]
        assert set(run.stages) == {"plan", "findings", "draft", "review"}
        assert run.llm_calls == 4
        assert run.latency_ms == 4 * MOCK_CALL_MS


def test_unknown_topic_scores_zero_and_empty_topic_raises():
    run = run_langgraph_research("The history of COBOL", llm=ResearchMockLLM())
    assert run.case_id == ""
    assert run.quality == 0.0
    assert "REJECTED" in run.final
    with pytest.raises(ValueError):
        run_crewai_research("   ")


def test_comparison_is_deterministic_and_prompt_sizes_differ():
    mock = ResearchMockLLM()
    first = compare_frameworks(llm=mock, backend="builtin")
    second = compare_frameworks(llm=ResearchMockLLM(), backend="builtin")
    signature = [
        (score.framework, score.quality, score.total_tokens, score.latency_ms, score.code_lines)
        for score in first
    ]
    assert signature == [
        (score.framework, score.quality, score.total_tokens, score.latency_ms, score.code_lines)
        for score in second
    ]
    by_name = {score.framework: score for score in first}
    assert set(by_name) == {"crewai", "langgraph", "agent_framework"}
    for score in first:
        assert score.quality == 1.0
        assert score.passed == score.cases == 3
        assert score.latency_ms == score.llm_calls * MOCK_CALL_MS
        assert score.code_lines == code_lines_for(score.framework)
        module = importlib.import_module(_MODULES[score.framework])
        assert score.code_lines == count_code_lines(Path(module.__file__))
    assert by_name["langgraph"].prompt_tokens < by_name["crewai"].prompt_tokens
    assert by_name["langgraph"].prompt_tokens < by_name["agent_framework"].prompt_tokens
    assert len({score.total_tokens for score in first}) == 3


def test_maf_async_entrypoint_and_running_loop():
    import asyncio

    from orchestrator.frameworks.maf_research import _run_library_sync, arun_maf_research

    async def _check() -> None:
        run = await arun_maf_research(
            "The history of Python", llm=ResearchMockLLM(), backend="builtin"
        )
        assert run.quality == 1.0
        with pytest.raises(RuntimeError, match="arun_maf_research"):
            _run_library_sync("The history of Python", None)

    asyncio.run(_check())


def test_research_mock_refuses_a_stage_without_the_previous_text():
    reply = ResearchMockLLM().complete(
        [
            Message(MessageRole.SYSTEM, "ROLE: researcher"),
            Message(MessageRole.USER, "Topic: The history of Python"),
        ]
    )
    assert "REJECTED" in reply.content
    planned = ResearchMockLLM().complete(
        [
            Message(MessageRole.SYSTEM, "ROLE: researcher"),
            Message(MessageRole.USER, "Topic: The history of Python\nRESEARCH PLAN:\n1. dates"),
        ]
    )
    assert "Guido van Rossum" in planned.content


def test_quality_score_requires_approval():
    case = load_cases()[0]
    assert quality_score("Guido van Rossum 1991", case) == 0.0
    assert quality_score("NOT APPROVED. Guido van Rossum 1991", case) == 0.0
    assert quality_score(case["review"], case) == 1.0
    original = case["topic"]
    case["topic"] = "changed by the caller"
    assert load_cases()[0]["topic"] == original


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_library_backend_requires_its_package_or_matches_the_builtin():
    expectations = [
        ("crewai", "crewai", run_crewai_research),
        ("langgraph", "langgraph", run_langgraph_research),
        ("agent_framework", "agent_framework", run_maf_research),
    ]
    topic = "The history of Python"
    builtin = {
        name: runner(topic, llm=ResearchMockLLM(), backend="builtin")
        for name, _module, runner in expectations
    }
    for name, module, runner in expectations:
        if importlib.util.find_spec(module) is None:
            with pytest.raises(ImportError):
                runner(topic, llm=ResearchMockLLM(), backend=name)
            continue
        library_backend = name
        run = runner(topic, llm=ResearchMockLLM(), backend=library_backend)
        assert run.runtime == library_backend
        assert run.final == builtin[name].final
        assert run.quality == 1.0
        assert run.llm_calls == 4
        if name != "crewai":
            # CrewAI wraps its own prompt, so its token total differs.
            # LangGraph and Agent Framework use the same messages as the builtin path.
            assert run.prompt_tokens == builtin[name].prompt_tokens
