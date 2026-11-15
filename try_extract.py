from pydantic import BaseModel

from agent_tools import ExtractionError, Extractor


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str


def replying(text):
    """A fake LLM that always answers with this text."""
    return lambda messages: {"choices": [{"message": {"role": "assistant", "content": text}}]}


good = 'Sure! Here is the data:\n```json\n{"vendor": "Acme", "total": 1200, "currency": "INR"}\n```\nHope that helps!'

ex = Extractor(replying(good))
print(ex.build_messages("Paid 1,200 INR to Acme on 3 Nov", Invoice)[0]["content"])  # the prompt
inv = ex.extract("Paid 1,200 INR to Acme on 3 Nov", Invoice)
print(repr(inv), "|", type(inv).__name__, inv.total, "|", inv.model_dump_json())

cases = {
    "no json": "I could not find any invoice.",
    "wrong type": '{"vendor": "Acme", "total": "lots", "currency": "INR"}',
    "think block": '<think>maybe {"x": 1}?</think>{"vendor": "Acme", "total": 5, "currency": "INR"}',
    "missing field": '{"vendor": "Acme", "total": 5}',
}
for label, reply in cases.items():
    try:
        print(label, "->", Extractor(replying(reply)).extract("...", Invoice))
    except ExtractionError as e:
        print(label, "-> ExtractionError:", e.message)
