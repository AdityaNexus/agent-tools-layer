from agent_tools import (ToolRegistry, parse_tool_calls , assistant_message , results_to_messages , get_text)


registry = ToolRegistry()

@registry.register
def add(a : int , b : int)->int:
    """add two numbers"""
    return a + b


def fake_llm(messages , tools):
    """"Plays the model: first asks for two calls , then answers"""
    if not any(m.get("role") == "tool" for m in messages):
        calls = [
            {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}},
            {"id": "c2", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}},
        ]
        return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls}}]}
    return {"choices": [{"message": {"role": "assistant", "content": "All done."}}]}


def run_agent(llm , registry , prompt , provider = "openai",max_steps = 5):
    messages = [{"role":"user","content": prompt}]
    for _ in range(max_steps):
        respnse = llm(messages , registry.schemas())
        calls = parse_tool_calls(respnse , provider)
        if not calls:
            return get_text(respnse , provider)
        messages.append(assistant_message(respnse , provider))
        result = registry.execute_all(calls)
        for r in result:
            print(f" {r.name}({r.call_id}) ok ={r.ok} -> {r.content}")
        messages.extend(results_to_messages(result , provider))
    return "stopped: max-steps reached"

print(run_agent(fake_llm , registry , "add 2 and 3"))
