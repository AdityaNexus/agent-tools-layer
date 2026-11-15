"""Shared test helpers: builders for OpenAI-style replies and a scripted fake LLM."""

import copy


def tool_call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def reply_with_calls(*calls):
    return {
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": list(calls)}}]
    }


def reply_with_text(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class ScriptedLLM:
    """A fake LLM. Returns the replies in order and records everything it was sent.

    It accepts both call shapes used in this project:
      llm(messages, tools)                      <- ToolAgent
      llm(messages, response_format=...)        <- Extractor in native mode
    """

    def __init__(self, *replies):
        self._replies = iter(replies)
        self.calls = []  # a COPY of the messages at each call (the agent keeps appending)
        self.tools = []  # the tools argument at each call
        self.formats = []  # the response_format argument at each call

    def __call__(self, messages, tools=None, response_format=None):
        self.calls.append(copy.deepcopy(messages))
        self.tools.append(tools)
        self.formats.append(response_format)
        return next(self._replies)
