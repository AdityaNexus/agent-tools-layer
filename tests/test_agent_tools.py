import pytest

from agent_tools import (
    Tool,ToolRegistry , ToolCall , 
    ToolValidationError , ToolExecutionError , parse_tool_calls,
    assistant_message , results_to_messages
)
from agent_tools.schema import ArgsModelBuilder

def get_weather(city : str , unit : str = "celsius")->str:
    """get the weather for the city"""
    return f"the weather in {city} is 25 degrees {unit}"

def test_required_and_optional_fields():
    schema = ArgsModelBuilder(get_weather,"m").build().model_json_schema()
    assert schema["required"] == ["city"]
    assert schema["properties"]["unit"]["default"] == "celsius" 


def test_missing_type_hint_raises():
    def bad(x):
        """Doc."""

    with pytest.raises(ValueError, match="type hint"):
        ArgsModelBuilder(bad, "m").build()


def test_extra_arguments_forbidden_in_schema():
    schema = ArgsModelBuilder(get_weather, "m").build().model_json_schema()
    assert schema["additionalProperties"] is False


def test_clean_schema_keeps_a_parameter_named_title():
    def f(title: str) -> str:
        """Doc."""

    schema = ArgsModelBuilder.clean_schema(ArgsModelBuilder(f, "m").build().model_json_schema())
    assert "title" in schema["properties"]            # the parameter survives
    assert "title" not in schema["properties"]["title"]  # the auto-generated label is gone


# ----------Tool -----------
def test_openai_schea_shape():
    s = Tool(get_weather).to_openai()
    assert s["type"] == "function"
    assert s["function"]["name"] == "get_weather"
    assert s["function"]["description"] == "get the weather for the city"
    assert s["function"]["parameters"]["required"] == ["city"]



def test_missing_docstring_raises():
    def bad(x: int):
        return x

    with pytest.raises(ValueError, match="docstring"):
        Tool(bad)


def test_invoke_with_json_string():
    assert Tool(get_weather).invoke('{"city": "Jaipur"}') == "the weather in Jaipur is 25 degrees celsius"


def test_invoke_with_dict():
    assert "fahrenheit" in Tool(get_weather).invoke({"city": "Delhi", "unit": "fahrenheit"})


def test_wrong_argument_name():
    with pytest.raises(ToolValidationError ) as e:
        Tool(get_weather).invoke({"cty":"jaipur"})
    assert "cty" in str(e.value.message)

def test_broken_json():
    with pytest.raises(ToolValidationError, match="valid JSON"):
        Tool(get_weather).invoke('{"city": ')


def test_arguments_must_be_an_object():
    with pytest.raises(ToolValidationError, match="JSON object"):
        Tool(get_weather).invoke('["a", "b"]')

def test_function_exception_is_wrapped():
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("kaput")

    with pytest.raises(ToolExecutionError) as e:
        Tool(boom).invoke("{}")
    assert "kaput" in e.value.message



#-----Registry-------

@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(get_weather)

    @r.register
    def add(a:int , b:int)->int:
        """add numbers"""
        return a + b

    return r 


def test_unknown_tool_lists_available(registry):
    res = registry.execute(ToolCall("1","nope","{}"))
    assert not res.ok
    assert "get_weather" in res.error and "add" in res.error 


def test_duplicate_name_rejected(registry):
    with pytest.raises(ValueError, match="already registered"):
        registry.register(get_weather)


#-----provider glue--------
def test_openai_roundtrip(registry):
    response = {
        "choices" : [{"message":{
            "role":"assistant","content" : None , "refusal" : None,
            "tool_calls" : [{"id":"c1","type":"function",
                             "function" : {"name":"add","arguments" : '{"a": 2, "b": 3}'}}],
        }}]}

    calls = parse_tool_calls(response)

    assert calls == [ToolCall("c1","add",'{"a": 2, "b": 3}')]

    assistant = assistant_message(response)
    msgs = results_to_messages(registry.execute_all(calls))
    assert msgs == [{"role": "tool", "tool_call_id": "c1", "content": "5"}]


def test_anthropic_roundtrip(registry):
    response = {"content": [
        {"type": "text", "text": "Calling add."},
        {"type": "tool_use", "id": "tu1", "name": "add", "input": {"a": 1, "b": 2}},
    ]}
    calls = parse_tool_calls(response, "anthropic")
    assert calls == [ToolCall("tu1", "add", {"a": 1, "b": 2})]

    msgs = results_to_messages(registry.execute_all(calls), "anthropic")
    assert msgs == [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tu1", "content": "3", "is_error": False}
    ]}]