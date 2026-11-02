
import json
from pydantic import Field
from agent_tools.tool import Tool
def get_weather(city : str , unit : str = "celsius")->str:
    """Get the weather for a city."""
    return f"28 degree {unit} in {city}"

t = Tool(get_weather)
print(json.dumps(t.to_openai(), indent=2))

# print(t.invoke('{"city": "New York"}'))
# print(t.invoke({"city": "New York", "unit": "fahrenheit"}))
# print(t.invoke({"cty":"jaipur"}))
t.invoke('{"city": ')



