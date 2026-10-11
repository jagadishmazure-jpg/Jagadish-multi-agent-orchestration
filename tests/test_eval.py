"""Tests for the trajectory recorder and task-success checks."""

import json

from orchestrator.core import TaskState
from orchestrator.eval import (
    TrajectoryRecorder,
    check_agents_involved,
    check_all_subtasks_done,
    check_contains,
    check_no_errors,
    check_step_budget,
)


def _recorder() -> TrajectoryRecorder:
    rec = TrajectoryRecorder()
    rec.record("task_start", agent="a", task="t")
    rec.record("step", agent="b")
    return rec


def test_recorder_seq_increments():
    rec = TrajectoryRecorder()
    e1 = rec.record("x", agent="a")
    e2 = rec.record("y", agent="b")
    assert e2.seq == e1.seq + 1
    assert len(rec) == 2


def test_recorder_filter():
    rec = _recorder()
    assert [e.kind for e in rec.filter("step")] == ["step"]


def test_recorder_agents_unique_ordered():
    rec = TrajectoryRecorder()
    rec.record("x", agent="b")
    rec.record("y", agent="a")
    rec.record("z", agent="b")
    assert rec.agents() == ["b", "a"]


def test_recorder_json_roundtrip():
    rec = _recorder()
    data = json.loads(rec.to_json())
    assert len(data["events"]) == 2
    assert data["events"][0]["kind"] == "task_start"


def test_check_contains_pass():
    result = check_contains("the quick brown fox", ["quick", "fox"])
    assert result.passed


def test_check_contains_fail():
    result = check_contains("the quick brown fox", ["quick", "zebra"])
    assert not result.passed
    assert "zebra" in result.detail


def test_check_all_subtasks_done_pass():
    assert check_all_subtasks_done({"a": "x", "b": "y"}).passed


def test_check_all_subtasks_done_fail():
    result = check_all_subtasks_done({"a": "x", "b": "  "})
    assert not result.passed
    assert "b" in result.detail


def test_check_no_errors_pass():
    assert check_no_errors(TaskState(task="t")).passed


def test_check_no_errors_fail():
    state = TaskState(task="t")
    state.mark_failed("boom")
    result = check_no_errors(state)
    assert not result.passed
    assert "boom" in result.detail


def test_check_step_budget_pass():
    assert check_step_budget(_recorder(), max_steps=5).passed


def test_check_step_budget_fail():
    assert not check_step_budget(_recorder(), max_steps=1).passed


def test_check_agents_involved_pass():
    assert check_agents_involved(_recorder(), ["a", "b"]).passed


def test_check_agents_involved_fail():
    result = check_agents_involved(_recorder(), ["a", "ghost"])
    assert not result.passed
    assert "ghost" in result.detail
