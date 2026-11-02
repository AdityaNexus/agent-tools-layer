import inspect
from pydantic import create_model , ConfigDict
from typing import get_type_hints

class ArgsModelBuilder:
    def __init__(self, func, model_name: str):
        self.func = func
        self.model_name = model_name

    def build(self):
        fields = {}
        hints = get_type_hints(self.func)
        for name, param in inspect.signature(self.func).parameters.items():
            if name not in hints:
                raise ValueError(f"Missing type hint for parameter '{name}' in function '{self.func.__name__}'")
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[name] = (hints[name], default)
        return create_model(self.model_name, **fields, __config__=ConfigDict(extra="forbid"))
  
    @staticmethod
    def clean_schema(node):
        """remove pydantic`s auto-generated "title" and "description" from the schema"""
        if isinstance(node , dict):
            return {
                K : ArgsModelBuilder.clean_schema(v)
                for K , v in node.items()
                if not (K == "title" and isinstance(v,str))
            }
        if isinstance(node , list):
            return [ArgsModelBuilder.clean_schema(v) for v in node]
        return node


    
