"""bundle everything one tool needs: the function, its name, its
 description, and its argument model. The class has three jobs: 
 describe (JSON for the LLM), validate (check what the LLM sent),
   run (call your function)."""

import inspect 
import json
from pydantic import ValidationError

from .schema import ArgsModelBuilder 

clean = ArgsModelBuilder.clean_schema
from .errors import ToolValidationError, ToolExecutionError

def _format_errors(exc : ValidationError)->str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "arguments"
        msg = err["msg"]
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts)


class Tool:
    def __init__(self , func , name = None , description = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func)
        if not self.description:
            raise ValueError(f"Function {self.name} has no docstring. Please provide a description.")
        self.args_model = ArgsModelBuilder(func , f"{self.name}_args").build()

    def to_openai(self):
        return {
            "type" : "function",
            "function" :{
                "name": self.name,
                "description": self.description,
                "parameters" : clean(self.args_model.model_json_schema()),
            },
        }

    def parse_arguments(self , raw):
        if isinstance(raw , str):
            try:
                data = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as e:
                raise ToolValidationError(self.name , f"Invalid JSON: {e.msg}") from e  # from e keeps the original traceback for you , while messae is the short text for the model
        else:
            data = raw
        if not isinstance(data , dict):
            raise ToolValidationError(self.name , f"Expected a JSON object but got {type(data).__name__}")

        try:
            model = self.args_model.model_validate(data)
        except ValidationError as e:
            raise ToolValidationError(self.name , _format_errors(e)) from e
        return model.model_dump()
    

    def invoke(self, raw):
        kwargs = self.parse_arguments(raw)
        try:
            return self.func(**kwargs)
        except Exception as e:
            raise ToolExecutionError(self.name , f"Function raised an exception: {e}") from e