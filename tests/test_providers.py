from agent_tools import (
    ToolCall,
    ToolResult,
    assistant_message,
    get_text,
    parse_tool_calls,
    results_to_messages,
)

OPENAI = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "refusal": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                    }
                ],
            }
        }
    ]
}

ANTHROPIC = {
    "content": [
        {"type": "text", "text": "Calling add."},
        {"type": "tool_use", "id": "tu1", "name": "add", "input": {"a": 1, "b": 2}},
    ]
}


def test_openai_tool_calls_are_parsed():
    assert parse_tool_calls(OPENAI) == [ToolCall("c1", "add", '{"a": 1, "b": 2}')]


def test_openai_missing_arguments_become_an_empty_object():
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "ping", "arguments": None},
                        }
                    ]
                }
            }
        ]
    }
    assert parse_tool_calls(response)[0].arguments == "{}"


def test_openai_reply_without_tool_calls_gives_an_empty_list():
    assert parse_tool_calls({"choices": [{"message": {"content": "hi"}}]}) == []


def test_openai_assistant_message_keeps_only_what_the_api_accepts_back():
    message = assistant_message(OPENAI)
    assert set(message) == {"role", "content", "tool_calls"}  # "refusal" is dropped


def test_openai_results_become_tool_messages():
    ok = ToolResult("c1", "add", ok=True, output=3)
    bad = ToolResult("c2", "add", ok=False, error="b: Field required")
    assert results_to_messages([ok, bad]) == [
        {"role": "tool", "tool_call_id": "c1", "content": "3"},
        {"role": "tool", "tool_call_id": "c2", "content": "Error: b: Field required"},
    ]


def test_anthropic_tool_calls_are_parsed():
    assert parse_tool_calls(ANTHROPIC, "anthropic") == [ToolCall("tu1", "add", {"a": 1, "b": 2})]


def test_anthropic_results_go_in_one_user_message():
    ok = ToolResult("tu1", "add", ok=True, output=3)
    bad = ToolResult("tu2", "add", ok=False, error="boom")
    assert results_to_messages([ok, bad], "anthropic") == [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu1", "content": "3", "is_error": False},
                {
                    "type": "tool_result",
                    "tool_use_id": "tu2",
                    "content": "Error: boom",
                    "is_error": True,
                },
            ],
        }
    ]


def test_get_text_for_both_providers():
    assert get_text({"choices": [{"message": {"content": "hello"}}]}) == "hello"
    assert get_text(OPENAI) == ""  # only tool calls, no text
    assert get_text(ANTHROPIC, "anthropic") == "Calling add."


def test_no_results_means_no_messages():
    assert results_to_messages([]) == []
    assert results_to_messages([], "anthropic") == []


def test_sdk_like_objects_are_accepted():
    class FakeSdkResponse:  # real SDK responses are Pydantic models like this
        def model_dump(self):
            return OPENAI

    assert parse_tool_calls(FakeSdkResponse())[0].name == "add"
