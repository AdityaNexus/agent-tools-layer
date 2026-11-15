import pytest
from helpers import ScriptedLLM, reply_with_calls, reply_with_text, tool_call

from agent_tools import ToolAgent, ToolRegistry


@pytest.fixture
def registry():
    r = ToolRegistry()

    @r.register
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    return r


GOOD = tool_call("c1", "add", '{"a": 2, "b": 3}')
BAD = tool_call("c2", "add", '{"a": 1}')  # missing b


def test_answers_directly_when_no_tool_is_needed(registry):
    run = ToolAgent(ScriptedLLM(reply_with_text("hello")), registry).run("hi")
    assert (run.status, run.steps, run.answer, run.tool_results) == ("completed", 1, "hello", [])


def test_tool_call_then_final_answer(registry):
    llm = ScriptedLLM(reply_with_calls(GOOD), reply_with_text("The sum is 5."))
    run = ToolAgent(llm, registry).run("add 2 and 3")
    assert run.status == "completed" and run.steps == 2 and run.answer == "The sum is 5."
    assert run.tool_results[0].output == 5
    assert [m["role"] for m in run.messages] == ["user", "assistant", "tool", "assistant"]


def test_the_model_is_sent_the_tool_schemas_every_step(registry):
    llm = ScriptedLLM(reply_with_calls(GOOD), reply_with_text("done"))
    ToolAgent(llm, registry).run("go")
    assert llm.tools == [registry.schemas(), registry.schemas()]


def test_the_model_sees_the_error_and_can_recover(registry):
    llm = ScriptedLLM(
        reply_with_calls(BAD),  # wrong call
        reply_with_calls(tool_call("c3", "add", '{"a": 1, "b": 1}')),  # corrected
        reply_with_text("2"),
    )
    run = ToolAgent(llm, registry).run("add 1 and 1")
    assert "b: Field required" in llm.calls[1][-1]["content"]  # the 2nd call saw the error text
    assert run.status == "completed" and run.steps == 3
    assert [r.ok for r in run.tool_results] == [False, True]


def test_stops_when_every_call_keeps_failing(registry):
    llm = ScriptedLLM(*[reply_with_calls(BAD)] * 5)
    run = ToolAgent(llm, registry).run("loop")
    assert (run.status, run.steps) == ("too_many_errors", 3)
    assert len(llm.calls) == 3  # it really stopped calling the model


def test_the_error_limit_is_configurable(registry):
    llm = ScriptedLLM(*[reply_with_calls(BAD)] * 5)
    run = ToolAgent(llm, registry, max_consecutive_errors=2).run("loop")
    assert (run.status, run.steps) == ("too_many_errors", 2)


def test_one_success_in_a_step_resets_the_error_counter(registry):
    llm = ScriptedLLM(
        reply_with_calls(BAD),  # step 1: all failed  -> counter 1
        reply_with_calls(GOOD, BAD),  # step 2: one worked  -> counter reset to 0
        reply_with_calls(BAD),  # step 3: all failed  -> counter 1 (not 2)
        reply_with_text("finished"),
    )
    run = ToolAgent(llm, registry, max_consecutive_errors=2).run("go")
    assert run.status == "completed" and run.steps == 4


def test_stops_at_max_steps_when_the_model_never_finishes(registry):
    llm = ScriptedLLM(*[reply_with_calls(GOOD)] * 5)
    run = ToolAgent(llm, registry, max_steps=3).run("forever")
    assert (run.status, run.steps) == ("max_steps", 3)


def test_a_message_list_can_be_the_prompt_and_is_not_modified(registry):
    prompt = [{"role": "user", "content": "hi"}]
    ToolAgent(ScriptedLLM(reply_with_text("hello")), registry).run(prompt)
    assert prompt == [{"role": "user", "content": "hi"}]
