import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .errors import ExtractionError
from .providers import get_text
from .schema import ArgsModelBuilder, formate_errors

T = TypeVar("T", bound=BaseModel)

_THINK = re.compile(r"<think>.*?</think>", re.S)
_decoder = json.JSONDecoder()

SYSTEM_PROMPT = (
    "You extract structured data from the user's text.\n"
    "Reply with ONE JSON object that matches the JSON Schema below, and nothing else.\n"
    "Only use information that is in the text. Never invent values.\n\n"
    "JSON Schema:\n{schema}"
)
RETRY_PROMPT = (
    "Your previous reply was rejected: {error}\n"
    "Reply again with ONE JSON object that matches the JSON Schema, and nothing else."
)


def extract_json(text: str) -> dict:
    """Find the first JSON object inside a messy model reply."""
    text = _THINK.sub("", text)
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _end = _decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            return obj
    raise ExtractionError("no JSON object found in the reply", text)


class Extractor:
    def __init__(self, llm, provider="openai", instructions=None, native_json=False):
        if native_json and provider == "anthropic":
            raise ValueError(
                "native_json needs an OpenAI-compatible server (Anthropic has no response_format)"
            )
        self.llm = llm
        self.provider = provider
        self.instructions = instructions
        self.native_json = native_json

    def _schema(self, model):
        return ArgsModelBuilder.clean_schema(model.model_json_schema())

    def build_messages(self, text: str, model: type[BaseModel]):
        system = SYSTEM_PROMPT.format(schema=json.dumps(self._schema(model), indent=2))
        if self.instructions:
            system += f"\n\nExtra instructions: {self.instructions}"
        if self.provider == "anthropic":
            return [{"role": "user", "content": f"{system}\n\nText:\n{text}"}]
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    def _call(self, messages, model):
        if self.native_json:
            fmt = {
                "type": "json_schema",
                "json_schema": {"name": model.__name__, "schema": self._schema(model)},
            }
            return self.llm(messages, response_format=fmt)
        return self.llm(messages)

    @staticmethod
    def _parse(raw: str, model: type[T]) -> T:
        data = extract_json(raw)
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise ExtractionError(formate_errors(exc), raw) from exc

    def extract(self, text: str, model: type[T], max_retries: int = 2) -> T:
        messages = self.build_messages(text, model)
        for attempt in range(1, max_retries + 2):  # first try + max_retries retries
            response = self._call(messages, model)
            raw = response if isinstance(response, str) else get_text(response, self.provider)
            try:
                return self._parse(raw, model)
            except ExtractionError as exc:
                if attempt > max_retries:
                    exc.attempts = attempt
                    raise
                shown = _THINK.sub("", raw).strip() or "(empty reply)"
                messages.append({"role": "assistant", "content": shown})
                messages.append({"role": "user", "content": RETRY_PROMPT.format(error=exc.message)})
