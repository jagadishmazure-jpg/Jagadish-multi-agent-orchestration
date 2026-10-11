"""Tests for the LLM client interface and FakeLLM."""

from orchestrator.core import Message, MessageRole
from orchestrator.llm import FakeLLM, LLMClient, LLMResponse, ToolCall


def _msgs(*texts: str) -> list[Message]:
    return [Message(MessageRole.USER, t) for t in texts]


def test_fake_llm_satisfies_interface():
    llm = FakeLLM()
    assert isinstance(llm, LLMClient)


def test_rule_substring_match():
    llm = FakeLLM(rules=[("hello", "hi there")])
    assert llm.complete(_msgs("say hello please")).content == "hi there"


def test_rule_case_insensitive():
    llm = FakeLLM(rules=[("HELLO", "hi")])
    assert llm.complete(_msgs("hello world")).content == "hi"


def test_default_when_no_rule_matches():
    llm = FakeLLM(rules=[("zzz", "matched")], default="fallback")
    assert llm.complete(_msgs("nothing relevant")).content == "fallback"


def test_first_matching_rule_wins():
    llm = FakeLLM(rules=[("a", "first"), ("a", "second")])
    assert llm.complete(_msgs("a")).content == "first"


def test_callable_rule():
    llm = FakeLLM(rules=[(lambda t: t.startswith("ping"), "pong")])
    assert llm.complete(_msgs("ping 123")).content == "pong"
    assert llm.complete(_msgs("nope")).content == "Acknowledged."


def test_callable_rule_exception_falls_through():
    def bad(text: str) -> bool:
        raise RuntimeError("boom")

    llm = FakeLLM(rules=[(bad, "x")], default="safe")
    assert llm.complete(_msgs("anything")).content == "safe"


def test_call_count_increments():
    llm = FakeLLM()
    llm.complete(_msgs("one"))
    llm.complete(_msgs("two"))
    assert llm.calls == 2


def test_history_recorded():
    llm = FakeLLM(rules=[("hi", "hello")])
    llm.complete(_msgs("hi"))
    assert llm.history[0][0] == "hi"
    assert llm.history[0][1].content == "hello"


def test_dict_response_with_tool_calls():
    llm = FakeLLM(
        rules=[
            (
                "calc",
                {
                    "content": "computing",
                    "tool_calls": [
                        {"name": "calculator", "arguments": {"expression": "1+1"}}
                    ],
                },
            )
        ]
    )
    resp = llm.complete(_msgs("please calc"))
    assert resp.content == "computing"
    assert len(resp.tool_calls) == 1
    assert isinstance(resp.tool_calls[0], ToolCall)
    assert resp.tool_calls[0].name == "calculator"
    assert resp.tool_calls[0].arguments == {"expression": "1+1"}


def test_llmresponse_passthrough():
    inner = LLMResponse(content="direct", model="custom")
    llm = FakeLLM(rules=[("x", inner)])
    resp = llm.complete(_msgs("x"))
    assert resp is inner


def test_sequence_in_order():
    llm = FakeLLM(sequence=["first", "second"])
    assert llm.complete(_msgs("a")).content == "first"
    assert llm.complete(_msgs("a")).content == "second"


def test_sequence_repeats_last_when_exhausted():
    llm = FakeLLM(sequence=["only"])
    llm.complete(_msgs("a"))
    assert llm.complete(_msgs("a")).content == "only"


def test_deterministic_same_input_same_output():
    llm = FakeLLM(rules=[("k", "v")], default="d")
    assert llm.complete(_msgs("k")).content == llm.complete(_msgs("k")).content


def test_add_rule():
    llm = FakeLLM()
    llm.add_rule("late", "added")
    assert llm.complete(_msgs("late arrival")).content == "added"


def test_empty_messages_uses_default():
    llm = FakeLLM(default="empty-ok")
    assert llm.complete([]).content == "empty-ok"


def test_model_name_recorded():
    llm = FakeLLM(name="test-model")
    assert llm.complete(_msgs("hi")).model == "test-model"
