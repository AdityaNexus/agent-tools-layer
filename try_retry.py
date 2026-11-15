from pydantic import BaseModel

from agent_tools import ExtractionError, Extractor


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str


GOOD = '{"vendor": "Acme", "total": 1200, "currency": "INR"}'
BAD = '{"vendor": "Acme", "total": "lots", "currency": "INR"}'


def scripted(*replies):
    """Fake LLM: returns the replies in order and records what it was sent."""
    it = iter(replies)
    seen = []

    def llm(messages, response_format=None):
        seen.append((len(messages), messages[-1]["content"][:70], response_format is not None))
        return next(it)

    llm.seen = seen
    return llm


print("--- 1: bad reply, then a good one")
llm = scripted(BAD, GOOD)
print(Extractor(llm).extract("Paid 1,200 INR to Acme", Invoice))
for s in llm.seen:
    print("  call:", s)

print("--- 2: always bad")
llm = scripted(BAD, BAD, BAD)
try:
    Extractor(llm).extract("...", Invoice)
except ExtractionError as e:
    print("  failed after", e.attempts, "attempts:", e.message)

print("--- 3: retries off")
try:
    Extractor(scripted(BAD)).extract("...", Invoice, max_retries=0)
except ExtractionError as e:
    print("  attempts:", e.attempts)

print("--- 4: native mode")
llm = scripted(GOOD)
Extractor(llm, native_json=True).extract("...", Invoice)
print("  response_format sent:", llm.seen[0][2])

try:
    Extractor(scripted(GOOD), provider="anthropic", native_json=True)
except ValueError as e:
    print("  ValueError:", e)
