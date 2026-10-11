"""Tests for messages, agents, task state, and the handoff protocol."""

import pytest

from orchestrator.core import Agent, Handoff, Message, MessageRole, TaskState, perform_handoff
from orchestrator.llm import FakeLLM


def test_message_defaults():
    msg = Message(MessageRole.USER, "hello")
    assert msg.sender == ""
    assert msg.metadata == {}
    assert isinstance(msg.seq, int)


def test_message_role_coerced_from_string():
    msg = Message("assistant", "hi")
    assert msg.role is MessageRole.ASSISTANT


def test_message_empty_content_allowed_for_tool_calls():
    # Tool-call-only assistant messages legitimately carry empty content.
    msg = Message(MessageRole.ASSISTANT, "", sender="bot")
    assert msg.content == ""


def test_message_non_string_content_raises():
    with pytest.raises(ValueError):
        Message(MessageRole.USER, None)  # type: ignore[arg-type]


def test_message_seq_increments():
    a = Message(MessageRole.USER, "a")
    b = Message(MessageRole.USER, "b")
    assert b.seq > a.seq


def test_message_short_truncates():
    msg = Message(MessageRole.USER, "x" * 200)
    assert len(msg.short()) <= 80
    assert msg.short().endswith("...")


def test_agent_respond_sets_sender(simple_agent: Agent):
    reply = simple_agent.respond([Message(MessageRole.USER, "hi")])
    assert reply.sender == "worker"
    assert reply.role is MessageRole.ASSISTANT
    assert reply.content == "ok"


def test_agent_default_system_prompt():
    agent = Agent(name="bob", role="planner", llm=FakeLLM())
    assert "bob" in agent.system_prompt and "planner" in agent.system_prompt


def test_agent_custom_system_prompt():
    agent = Agent(name="bob", role="planner", llm=FakeLLM(), system_prompt="Be brief.")
    assert agent.system_prompt == "Be brief."


def test_agent_empty_name_raises():
    with pytest.raises(ValueError):
        Agent(name="", role="r", llm=FakeLLM())


def test_agent_tool_schemas_empty_by_default(simple_agent: Agent):
    assert simple_agent.tool_schemas() == []


def test_task_state_add_message(task_state: TaskState):
    msg = Message(MessageRole.USER, "hi")
    task_state.add_message(msg)
    assert task_state.messages == [msg]


def test_task_state_artifacts_roundtrip(task_state: TaskState):
    task_state.set_artifact("plan", [1, 2])
    assert task_state.get_artifact("plan") == [1, 2]


def test_task_state_artifact_default(task_state: TaskState):
    assert task_state.get_artifact("missing", "dflt") == "dflt"


def test_task_state_advance_and_status(task_state: TaskState):
    task_state.advance()
    assert task_state.step == 1
    task_state.mark_done()
    assert task_state.status == "done"
    assert task_state.ok


def test_task_state_mark_failed(task_state: TaskState):
    task_state.mark_failed("bad thing")
    assert task_state.status == "failed"
    assert not task_state.ok
    assert task_state.errors == ["bad thing"]


def test_handoff_describe_contains_parties():
    h = Handoff(from_agent="a", to_agent="b", payload={"input": "x"}, reason="next")
    text = h.describe()
    assert "a" in text and "b" in text and "next" in text and "x" in text


def test_handoff_validate_missing_to_raises():
    with pytest.raises(ValueError):
        Handoff(from_agent="a", to_agent="").validate()


def test_handoff_validate_self_raises():
    with pytest.raises(ValueError):
        Handoff(from_agent="a", to_agent="a").validate()


def test_perform_handoff_records_message(task_state: TaskState, simple_agent: Agent):
    agents = {"worker": simple_agent}
    h = Handoff(from_agent="pipeline", to_agent="worker", payload={"input": "do it"})
    msg = perform_handoff(h, agents, task_state)
    assert msg.recipient == "worker"
    assert msg.metadata["handoff"] is True
    assert task_state.messages[-1] is msg
    assert task_state.step == 1


def test_perform_handoff_unknown_target_raises(task_state: TaskState):
    h = Handoff(from_agent="a", to_agent="ghost", payload={})
    with pytest.raises(KeyError):
        perform_handoff(h, {}, task_state)
