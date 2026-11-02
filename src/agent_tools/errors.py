class ToolError(Exception):
    """"Base class. carries the tool name and a message safe to show the LLM"""
    
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"Tool '{tool_name}' error: {message}")


class ToolValidationError(ToolError):
    """"the LLM`s arguments are broken JSON or dont`t match the schema"""


class ToolExecutionError(ToolError):
    """"the tool function raised an exception. The message is safe to show the LLM"""