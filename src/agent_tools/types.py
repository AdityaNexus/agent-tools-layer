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
    def to_message(self,provider:str = "openai")->dict:
        if provider == "anthropic":
            return {
                "type" : "tool_result" ,
                "tool_use_id" : self.call_id,
                "content" : self.content,
                "is_error" : not self.ok,
            }
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "content": self.content,
        }