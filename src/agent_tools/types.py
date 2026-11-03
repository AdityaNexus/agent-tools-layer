import json 
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """one tool request from the LLM"""
    id : str
    name : str
    arguments : Any = field(default_factory=dict)



@dataclass
class ToolResult:
    """"What happend when we ran a ToolCall. Errors are data , not exceptions"""
    call_id : str
    name : str
    ok : bool
    output : Any = field(default_factory=dict)
    error: str | None = None

    @property
    def content(self)->str:
        """The text the LLM will read"""
        if not self.ok:
            return f"Error : {self.error}"
        if isinstance(self.output, str):
            return self.output
        return json.dumps(self.output,default = str)