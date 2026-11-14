import inspect
from pydantic import create_model , ConfigDict ,Field
from pydantic.fields import FieldInfo
from typing import get_type_hints
import re
class ArgsModelBuilder:
    def __init__(self, func, model_name: str, doc_params=None):
        self.func = func
        self.model_name = model_name
        self.doc_params = doc_params or {}

    def build(self):
        fname = getattr(self.func, "__name__", self.model_name)
        hints = get_type_hints(self.func, include_extras=True)   # keep Annotated[...] metadata
        fields = {}
        for name, param in inspect.signature(self.func).parameters.items():
            if name not in hints:
                raise ValueError(f"Missing type hint for parameter '{name}' in function '{fname}'")
            default = ... if param.default is inspect.Parameter.empty else param.default
            description = self.doc_params.get(name)
            if description and not self._has_description(hints[name]):
                default = Field(default, description=description)   # ... still means required
            fields[name] = (hints[name], default)
        return create_model(self.model_name, **fields, __config__=ConfigDict(extra="forbid"))

    @staticmethod
    def _has_description(annotation):
        meta = getattr(annotation, "__metadata__", ())
        return any(isinstance(m, FieldInfo) and m.description for m in meta)

    @staticmethod
    def clean_schema(node):
        """Remove pydantic's auto-generated "title" entries (descriptions are kept)."""
        if isinstance(node, dict):
            return {
                k: ArgsModelBuilder.clean_schema(v)
                for k, v in node.items()
                if not (k == "title" and isinstance(v, str))
            }
        if isinstance(node, list):
            return [ArgsModelBuilder.clean_schema(v) for v in node]
        return node


    
_SECTION = re.compile(
    r"^(Args|Arguments|Parameters|Returns?|Raises?|Yields?|Examples?|Notes?)\s*:\s*$", re.I
)
_PARAM = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def parse_docstring(doc):
    """Split a Google-style docstring into (summary, {param_name: description})."""
    if not doc:
        return "", {}
    summary, params = [], {}
    state, current = "summary", None          # state: summary -> args / other
    for line in inspect.cleandoc(doc).splitlines():
        text = line.strip()
        section = _SECTION.match(text)
        if section:
            state = "args" if section.group(1).lower() in ("args", "arguments", "parameters") else "other"
            current = None
            continue
        if state == "summary":
            summary.append(text)
        elif state == "args" and text:
            m = _PARAM.match(text)
            if m:
                current = m.group(1)
                params[current] = m.group(2).strip()
            elif current:                      # continuation line of the previous parameter
                params[current] += " " + text
    return " ".join(s for s in summary if s), params