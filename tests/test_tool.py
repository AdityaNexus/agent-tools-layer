import pytest
from pydantic import BaseModel, Field

from agent_tools import Tool, ToolExecutionError, ToolValidationError


def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the weather for a city."""
    return f"28 degrees {unit} in {city}"


def test_openai_schema_shape():
    schema = Tool(get_weather).to_openai()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "get_weather"
    assert fn["description"] == "Get the weather for a city."
    assert fn["parameters"]["required"] == ["city"]


def test_description_is_the_summary_only():
    def add(a: int, b: int) -> int:
        """Add numbers.

        Args:
            a: first number
            b: second number
        """

    tool = Tool(add)
    assert tool.description == "Add numbers."
    assert (
        tool.to_openai()["function"]["parameters"]["properties"]["a"]["description"]
        == "first number"
    )


def test_missing_docstring_raises():
    def bad(x: int):
        return x

    with pytest.raises(ValueError, match="docstring"):
        Tool(bad)


def test_explicit_description_overrides_docstring():
    assert Tool(get_weather, description="Custom.").description == "Custom."


def test_invoke_with_json_string():
    assert Tool(get_weather).invoke('{"city": "Jaipur"}') == "28 degrees celsius in Jaipur"


def test_invoke_with_dict():
    assert "fahrenheit" in Tool(get_weather).invoke({"city": "Delhi", "unit": "fahrenheit"})


def test_empty_arguments_for_a_tool_without_parameters():
    def ping() -> str:
        """Ping."""
        return "pong"

    assert Tool(ping).invoke("") == "pong"
    assert Tool(ping).invoke("{}") == "pong"


def test_numeric_string_is_converted():
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    assert Tool(add).invoke({"a": "2", "b": 3}) == 5


def test_wrong_argument_name():
    with pytest.raises(ToolValidationError) as info:
        Tool(get_weather).invoke({"cty": "Jaipur"})
    assert "cty" in info.value.message and "city" in info.value.message


def test_wrong_type():
    with pytest.raises(ToolValidationError) as info:
        Tool(get_weather).invoke({"city": 5})
    assert info.value.message.startswith("city:")


def test_broken_json():
    with pytest.raises(ToolValidationError, match="Invalid JSON"):
        Tool(get_weather).invoke('{"city": ')


def test_arguments_must_be_an_object():
    with pytest.raises(ToolValidationError, match="Expected a JSON object"):
        Tool(get_weather).invoke('["a", "b"]')


def test_function_exception_keeps_its_type_name():
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("kaput")

    with pytest.raises(ToolExecutionError) as info:
        Tool(boom).invoke("{}")
    assert info.value.message == "RuntimeError: kaput"


def test_nested_model_arrives_as_an_object_not_a_dict():
    class Event(BaseModel):
        title: str
        hour: int = Field(ge=0, le=23)

    def schedule(event: Event) -> str:
        """Schedule."""
        assert isinstance(event, Event)
        return f"{event.title}@{event.hour}"

    tool = Tool(schedule)
    assert tool.invoke({"event": {"title": "Demo", "hour": 9}}) == "Demo@9"
    with pytest.raises(ToolValidationError) as info:
        tool.invoke({"event": {"title": "Demo", "hour": 30}})
    assert "event.hour" in info.value.message


def test_bound_method_hides_self():
    class Service:
        def ping(self, msg: str) -> str:
            """Echo."""
            return msg

    tool = Tool(Service().ping)
    params = tool.to_openai()["function"]["parameters"]
    assert list(params["properties"]) == ["msg"]
    assert tool.invoke({"msg": "hi"}) == "hi"
