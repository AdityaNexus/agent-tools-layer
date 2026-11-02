"""bundle everything one tool needs: the function, its name, its
 description, and its argument model. The class has three jobs: 
 describe (JSON for the LLM), validate (check what the LLM sent),
   run (call your function)."""

import inspect 
import json
from pydantic import ValidationError

from .schema import ArgsModelBuilder 

clean = ArgsModelBuilder.clean_schema


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
        data = json.loads(raw) if isinstance(raw , str) else raw
        model = self.args_model.model_validate(data)
        return model.model_dump()

    def invoke(self, raw):
        kwargs = self.parse_arguments(raw)
        return self.func(**kwargs)