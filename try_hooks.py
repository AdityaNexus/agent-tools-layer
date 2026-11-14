import logging
from agent_tools import ConfirmationHook , LoggingHook,ToolCall , ToolRegistry

logging.basicConfig(level=logging.INFO,format="%(name)s | %(message)s")

registry = ToolRegistry(hooks =[
    LoggingHook(),
    ConfirmationHook(ask = lambda tool , kwargs : kwargs["path"]!="/important"),
])


@registry.register
def add(a: int , b: int)->int:
    """add two numbers"""
    return a + b


@registry.register(requires_confirmation=True)
def delete_file(path: str) -> str:
    """Delete a file."""
    return f"deleted {path}"


calls = [
    ToolCall("1", "add", {"a": 1, "b": 2}),
    ToolCall("2", "delete_file", {"path": "/tmp/x"}),
    ToolCall("3", "delete_file", {"path": "/important"}),
]
for r in registry.execute_all(calls):
    print(r.call_id, r.ok, r.content, f"{r.duration_ms:.3f}ms")



unsafe = ToolRegistry()
unsafe.register(delete_file, requires_confirmation=True)
print(unsafe.execute(ToolCall("4", "delete_file", {"path": "/tmp/x"})).content)