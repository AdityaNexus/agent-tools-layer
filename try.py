
import json
from agent_tools import ToolRegistry , ToolCall

registry = ToolRegistry()
def get_weather(city:str , unit:str = "c")->str:
    """"get the weather for the city"""
    return f"The weather in {city} is 25 degrees {unit}"


@registry.register
def add(a:int , b:int)->int:
    """add two numbers"""
    return a + b

print(json.dumps(registry.schemas(), indent=2))



calls = [
    ToolCall("c1", "get_weather", '{"city": "Jaipur"}'),
    ToolCall("c2", "add", {"a": "2", "b": 3}),
    ToolCall("c3", "add", {"a": 1, "c": 3}),
    ToolCall("c4", "multiply", "{}"),
]
for r in registry.execute_all(calls):
    print(r.call_id, r.ok, "->", r.content)


