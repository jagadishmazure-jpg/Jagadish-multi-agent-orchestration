"""End-to-end tests for the two offline scenarios."""

import pytest

from orchestrator.eval import (
    check_agents_involved,
    check_contains,
    check_no_errors,
    check_step_budget,
)
from orchestrator.patterns import UnknownRouteError
from scenarios import run_research_scenario, triage_ticket


def test_research_team_final_contains_keywords():
    result = run_research_scenario()
    check = check_contains(result["final"], ["Python", "Guido", "1991"])
    assert check.passed, check.detail


def test_research_team_agents_involved():
    result = run_research_scenario()
    check = check_agents_involved(
        result["trajectory"], ["planner", "researcher", "writer", "critic"]
    )
    assert check.passed, check.detail


def test_research_team_deterministic():
    first = run_research_scenario()["final"]
    second = run_research_scenario()["final"]
    assert first == second


def test_research_team_step_budget():
    result = run_research_scenario()
    assert check_step_budget(result["trajectory"], max_steps=12).passed


def test_research_team_outputs_chain():
    result = run_research_scenario()
    assert len(result["outputs"]) == 4
    assert "PLAN" in result["outputs"][0]
    assert "APPROVED" in result["final"]


def test_triage_billing_routes_and_computes():
    result = triage_ticket("I was double billed, please refund me")
    assert result["category"] == "billing"
    assert result["agent"] == "billing_specialist"
    assert "89.97" in result["output"]


def test_triage_technical_routes_and_searches():
    result = triage_ticket("login error after update")
    assert result["category"] == "technical"
    assert "KB-1042" in result["output"]


def test_triage_account_routes():
    result = triage_ticket("need to change my account email")
    assert result["category"] == "account"
    assert "confirmation link" in result["output"]


def test_triage_unknown_category_raises():
    with pytest.raises(UnknownRouteError):
        triage_ticket("what is the meaning of life")


def test_triage_deterministic():
    first = triage_ticket("refund please")["output"]
    second = triage_ticket("refund please")["output"]
    assert first == second


def test_triage_no_errors():
    from orchestrator.core import TaskState

    result = triage_ticket("bill question")
    assert check_no_errors(TaskState(task=result["ticket"])).passed
