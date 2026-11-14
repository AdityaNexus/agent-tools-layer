from .errors import ToolError , ToolPermissionError
from .tool import Tool
from .types import ToolCall, ToolResult
import time
from .hooks import ConfirmationHook
import inspect
class ToolRegistry:
    def __init__(self,hooks = None):
        self._tools : dict[str,Tool] = {}
        self.hooks = hooks or []

    def add(self , tool_or_func):
        tool = tool_or_func if isinstance(tool_or_func , Tool) else Tool(tool_or_func)
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        return tool

    def register(self , func = None , * , requires_confirmation: bool = False):
        """"Use as @registry.register or @registry.register(requires_confirmation=True)"""
        def decorator(f):
            self.add(Tool(f, requires_confirmation=requires_confirmation))
            return f
        return decorator(func) if func is not None else decorator

    def schemas(self):
        return [t.to_openai() for t in self._tools.values()]

    def execute(self,call):
        tool = None
        duration = 0.0
        try:
            tool = self._tools.get(call.name)
            if tool is None:
                availabel = ", ".join(self._tools) or "None"
                raise ToolError(call.name , f"unknown tool . Available: {availabel}")
            kwargs = tool.parse_arguments(call.arguments)
            self._check_confirmation_is_configured(tool)
            for hook in self.hooks:
               kwargs = hook.before_call(tool, kwargs)

            started = time.perf_counter()
            try:
                output = tool.call(kwargs)
            finally:
                duration = (time.perf_counter() - started) * 1000.0
            result = ToolResult(
                call_id = call.id,
                name = call.name,
                ok = True,
                output = output
            )

        except ToolError as e:
            result = ToolResult(
                call_id = call.id,
                name = call.name,
                ok = False,
                error = e.message
            )
        result.duration_ms = duration
        if tool is not None:
            for hook in self.hooks:
                result = hook.after_call(tool, result)
        return result

    def _check_confirmation_is_configured(self, tool):
        if tool.requires_confirmation and not any(isinstance(h, ConfirmationHook) for h in self.hooks):
            raise ToolPermissionError(
                tool.name, "requires confirmation, but no ConfirmationHook is configured"
            )



    def execute_all(self , calls):
        return [self.execute(call) for call in calls]

    def add_object(self,obj,prefix = "",include =None , confirm = ()):
        """Register the documented public methods of an object as tools"""
        added = []
        for name , _ in inspect.getmembers(type(obj),inspect.isfunction):
            if name.startswith("_") or (include is not None and name not in include):
                continue
            method = getattr(obj,name)
            if inspect.getdoc(method):
                added.append(self.add(Tool(method , name = f"{prefix}{name}" , requires_confirmation = name in confirm)))
        return added

    