import pytest
from helpers import ScriptedLLM, reply_with_text
from pydantic import BaseModel

from agent_tools import ExtractionError, Extractor, extract_json


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str


GOOD = '{"vendor": "Acme", "total": 1200, "currency": "INR"}'
BAD = '{"vendor": "Acme", "total": "lots", "currency": "INR"}'
EXPECTED = Invoice(vendor="Acme", total=1200.0, currency="INR")


# ---------- extract_json ----------
@pytest.mark.parametrize(
    "reply",
    [
        '{"a": 1}',
        'Sure! Here you go:\n```json\n{"a": 1}\n```\nHope that helps.',
        '<think>maybe {"x": 1}</think>{"a": 1}',  # the decoy inside <think> is ignored
        'Use {braces} like this: {"a": 1}',  # a stray "{" that is not JSON
    ],
)
def test_extract_json_finds_the_object(reply):
    assert extract_json(reply) == {"a": 1}


def test_extract_json_returns_the_outer_object():
    assert extract_json('x {"a": {"b": 2}} y') == {"a": {"b": 2}}


@pytest.mark.parametrize("reply", ["", "no data here", "[1, 2, 3]", '{"cut": off'])
def test_extract_json_raises_when_there_is_no_object(reply):
    with pytest.raises(ExtractionError, match="no JSON object"):
        extract_json(reply)


# ---------- building the prompt ----------
def test_openai_prompt_has_a_system_message_with_the_schema():
    messages = Extractor(ScriptedLLM()).build_messages("Paid 1,200 INR", Invoice)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert '"vendor"' in messages[0]["content"] and '"required"' in messages[0]["content"]
    assert messages[1]["content"] == "Paid 1,200 INR"


def test_anthropic_prompt_is_one_user_message():
    messages = Extractor(ScriptedLLM(), provider="anthropic").build_messages("Paid", Invoice)
    assert [m["role"] for m in messages] == ["user"]
    assert "Paid" in messages[0]["content"] and '"vendor"' in messages[0]["content"]


def test_extra_instructions_are_added():
    ex = Extractor(ScriptedLLM(), instructions="Amounts are in rupees.")
    assert "Amounts are in rupees." in ex.build_messages("x", Invoice)[0]["content"]


# ---------- extracting ----------
def test_extract_returns_a_typed_object():
    result = Extractor(ScriptedLLM(reply_with_text(GOOD))).extract(
        "Paid 1,200 INR to Acme", Invoice
    )
    assert result == EXPECTED and isinstance(result, Invoice)
    assert result.total == 1200.0  # int in the JSON, float in the object


def test_a_plain_string_reply_is_accepted_too():
    assert Extractor(ScriptedLLM(GOOD)).extract("x", Invoice) == EXPECTED


def test_missing_fields_are_reported():
    with pytest.raises(ExtractionError) as info:
        Extractor(ScriptedLLM('{"vendor": "Acme"}')).extract("x", Invoice, max_retries=0)
    message = info.value.message
    assert "total: Field required" in message and "currency: Field required" in message


def test_optional_fields_may_be_absent():
    class Contact(BaseModel):
        name: str
        phone: str | None = None

    assert Extractor(ScriptedLLM('{"name": "Asha"}')).extract("x", Contact).phone is None


# ---------- retry with feedback ----------
def test_a_bad_reply_is_retried_with_the_error_as_feedback():
    llm = ScriptedLLM(BAD, GOOD)
    assert Extractor(llm).extract("x", Invoice) == EXPECTED
    assert [len(c) for c in llm.calls] == [2, 4]  # the history grows by 2 per failure
    assert llm.calls[1][2] == {"role": "assistant", "content": BAD}
    feedback = llm.calls[1][3]
    assert feedback["role"] == "user"
    assert "total: Input should be a valid number" in feedback["content"]


def test_it_recovers_from_a_reply_without_any_json():
    llm = ScriptedLLM("I cannot do that.", GOOD)
    assert Extractor(llm).extract("x", Invoice) == EXPECTED
    assert "no JSON object found" in llm.calls[1][3]["content"]


def test_it_gives_up_after_max_retries():
    llm = ScriptedLLM(BAD, BAD, BAD, BAD)
    with pytest.raises(ExtractionError) as info:
        Extractor(llm).extract("x", Invoice)
    assert info.value.attempts == 3 and len(llm.calls) == 3
    assert info.value.raw == BAD


def test_retries_can_be_switched_off():
    llm = ScriptedLLM(BAD, GOOD)
    with pytest.raises(ExtractionError) as info:
        Extractor(llm).extract("x", Invoice, max_retries=0)
    assert info.value.attempts == 1 and len(llm.calls) == 1


def test_think_blocks_are_not_kept_in_the_retry_history():
    llm = ScriptedLLM(f"<think>long reasoning...</think>{BAD}", GOOD)
    Extractor(llm).extract("x", Invoice)
    assert llm.calls[1][2]["content"] == BAD


def test_an_empty_reply_is_shown_as_such_in_the_history():
    llm = ScriptedLLM("", GOOD)
    Extractor(llm).extract("x", Invoice)
    assert llm.calls[1][2]["content"] == "(empty reply)"


# ---------- native JSON mode ----------
def test_native_mode_sends_the_schema_as_response_format():
    llm = ScriptedLLM(GOOD)
    Extractor(llm, native_json=True).extract("x", Invoice)
    fmt = llm.formats[0]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "Invoice"
    assert fmt["json_schema"]["schema"]["required"] == ["vendor", "total", "currency"]


def test_normal_mode_sends_no_response_format():
    llm = ScriptedLLM(GOOD)
    Extractor(llm).extract("x", Invoice)
    assert llm.formats == [None]


def test_native_mode_still_validates_and_retries():
    llm = ScriptedLLM(BAD, GOOD)
    assert Extractor(llm, native_json=True).extract("x", Invoice) == EXPECTED
    assert len(llm.calls) == 2


def test_native_mode_is_rejected_for_anthropic():
    with pytest.raises(ValueError, match="native_json"):
        Extractor(ScriptedLLM(), provider="anthropic", native_json=True)
