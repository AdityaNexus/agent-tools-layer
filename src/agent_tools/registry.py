from .errors import ToolError
from .tool import Tool
from .types import ToolCall, ToolResult


class ToolRegistry:
    def __init__(self):
        self._tools : dict[str,Tool] = {}


    def add(self , tool_or_func):
        tool = tool_or_func if isinstance(tool_or_func , Tool) else Tool(tool_or_func)
        if tool.name in self._tools:
            raise ToolError( f"Tool with name '{tool.name}' already exists in the registry")
        self._tools[tool.name] = tool
        return tool

    def register(self , func):
        """"Decorator to register a function as a tool in the registry"""
        self.add(func)
        return func

    def schemas(self):
        return [t.to_openai() for t in self._tools.values()]

    def execute(self , call : ToolCall)->ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self._tools) or "none"
            return ToolResult(
                call.id ,
                call.name,
                ok = False,
                error = f"Tool '{call.name}' not found. Available tools: {available}"

            )

        try:
            output = tool.invoke(call.arguments)
        except ToolError as e:
            return ToolResult(
                call.id,
                call.name,
                ok = False,
                error  = f"Tool '{call.name}' failed with error: {str(e)}"
            )
        return ToolResult(
            call.id,
            call.name,
            ok = True,
            output = output
        )


    def execute_all(self , calls):
        return [self.execute(call) for call in calls]