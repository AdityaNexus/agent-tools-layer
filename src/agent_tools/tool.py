"""bundle everything one tool needs: the function, its name, its
description, and its argument model. The class has three jobs:
describe (JSON for the LLM), validate (check what the LLM sent),
  run (call your function)."""

import inspect
import json

from pydantic import ValidationError

from .schema import ArgsModelBuilder, formate_errors, parse_docstring

clean = ArgsModelBuilder.clean_schema
from .errors import ToolExecutionError, ToolValidationError


class Tool:
    """A tool is a function with a name, description, and argument model."""

    def __init__(self, func, name=None, description=None, requires_confirmation: bool = False):
        self.func = func
        self.name = name or func.__name__
        self.requires_confirmation = requires_confirmation
        summary, doc_params = parse_docstring(inspect.getdoc(func))
        self.description = description or summary
        if not self.description:
            raise ValueError(
                f"Function {self.name} has no docstring. Please provide a description."
            )
        self.args_model = ArgsModelBuilder(func, f"{self.name}_args", doc_params).build()

    def to_openai(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": clean(self.args_model.model_json_schema()),
            },
        }

    def parse_arguments(self, raw):
        if isinstance(raw, str):
            try:
                data = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as e:
                raise ToolValidationError(
                    self.name, f"Invalid JSON: {e.msg}"
                ) from e  # from e keeps the original traceback for you , while messae is the short text for the model
        else:
            data = raw
        if not isinstance(data, dict):
            raise ToolValidationError(
                self.name, f"Expected a JSON object but got {type(data).__name__}"
            )

        try:
            model = self.args_model.model_validate(data)
        except ValidationError as e:
            raise ToolValidationError(self.name, formate_errors(e)) from e
        return {name: getattr(model, name) for name in type(model).model_fields}

    def invoke(self, raw):
        return self.call(self.parse_arguments(raw))

    def call(self, kwargs):
        try:
            return self.func(**kwargs)
        except Exception as e:
            raise ToolExecutionError(self.name, f"{type(e).__name__}: {e}") from e
