class ToolError(Exception):
    """ "Base class. carries the tool name and a message safe to show the LLM"""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"Tool '{tool_name}' error: {message}")


class ToolValidationError(ToolError):
    """ "the LLM`s arguments are broken JSON or dont`t match the schema"""


class ToolExecutionError(ToolError):
    """ "the tool function raised an exception. The message is safe to show the LLM"""


class ToolPermissionError(ToolError):
    """ "the tool function raised an exception. The message is safe to show the LLM"""


class ExtractionError(Exception):
    """The model`s could not be turned into the requested object"""

    def __init__(self, message: str, raw: str = "", attempts: int = 1):
        self.message = message  # short, safe to show the model on a retry
        self.raw = raw  # the model's last reply, for debugging
        self.attempts = attempts  # how many LLM calls were made
        super().__init__(message)
