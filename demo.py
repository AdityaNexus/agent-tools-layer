from agent_tools import ToolAgent, ToolRegistry

registry = ToolRegistry()


@registry.register
def add(a: int, b: int) -> int:
    """add two numbers"""
    return a + b


def fake_llm(messages, tools):
    """ "Plays the model: first asks for two calls , then answers"""
    if not any(m.get("role") == "tool" for m in messages):
        calls = [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
            },
            {"id": "c2", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}},
        ]
        return {
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls}}]
        }
    return {"choices": [{"message": {"role": "assistant", "content": "All done."}}]}


agent = ToolAgent(fake_llm, registry)
run = agent.run("add 2 and 3")
print(run.status, "|", run.steps, "steps |", run.answer)
print([(r.name, r.ok) for r in run.tool_results])


def always_bad_llm(messages, tools):
    calls = [{"id": "x", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}}]
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls}}]}


bad = ToolAgent(always_bad_llm, registry).run("loop forever")
print(bad.status, "|", bad.steps, "steps")
