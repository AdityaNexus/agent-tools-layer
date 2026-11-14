from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Any

from .errors import ToolPermissionError

if TYPE_CHECKING:
    from .tool import Tool
    from .types import ToolResult

logger = logging.getLogger("agent_tools")


class Hook:
    """Base class. Override only the methods you need."""

    def before_call(self, tool: Tool, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Runs after validation, before the function. Return kwargs (changed or not),
        or raise ToolPermissionError to block the call."""
        return kwargs

    def after_call(self, tool: Tool, result: ToolResult) -> ToolResult:
        """Runs after the function (also after failures). Return the result, changed or not."""
        return result


class ConfirmationHook(Hook):
    """Human in the loop: ask before running tools marked requires_confirmation."""

    def __init__(self, ask: Callable[[Tool, dict[str, Any]], bool] | None = None) -> None:
        self.ask = ask or self._ask_in_terminal

    def before_call(self, tool: Tool, kwargs: dict[str, Any]) -> dict[str, Any]:
        if tool.requires_confirmation and not self.ask(tool, kwargs):
            raise ToolPermissionError(tool.name, "the user did not approve this call")
        return kwargs

    @staticmethod
    def _ask_in_terminal(tool: Tool, kwargs: dict[str, Any]) -> bool:
        answer = input(f"Allow {tool.name}({kwargs})? [y/N] ")
        return answer.strip().lower() == "y"



class LoggingHook(Hook):
    def before_call(self, tool: Tool, kwargs: dict[str, Any]) -> dict[str, Any]:
        logger.info("calling %s %s", tool.name, kwargs)
        return kwargs

    def after_call(self, tool: Tool, result: ToolResult) -> ToolResult:
        logger.info("%s ok=%s %.1fms", tool.name, result.ok, result.duration_ms)
        return result