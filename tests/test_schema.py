from typing import Annotated

import pytest
from pydantic import Field, ValidationError

from agent_tools.schema import ArgsModelBuilder, formate_errors, parse_docstring


def weather(city: str, unit: str = "celsius") -> str:
    """Get the weather."""


def schema_of(func, doc_params=None):
    model = ArgsModelBuilder(func, "m", doc_params).build()
    return ArgsModelBuilder.clean_schema(model.model_json_schema())


def test_required_and_optional_fields():
    s = schema_of(weather)
    assert s["required"] == ["city"]
    assert s["properties"]["unit"]["default"] == "celsius"


def test_missing_type_hint_raises():
    def bad(x):
        """Doc."""

    with pytest.raises(ValueError, match="type hint"):
        ArgsModelBuilder(bad, "m").build()


def test_unknown_arguments_are_rejected():
    model = ArgsModelBuilder(weather, "m").build()
    with pytest.raises(ValidationError):
        model(city="Jaipur", cty="typo")
    assert schema_of(weather)["additionalProperties"] is False


def test_clean_schema_removes_titles_but_keeps_a_field_named_title():
    def doc(title: str, pages: int) -> None:
        """Doc."""

    raw = ArgsModelBuilder(doc, "m").build().model_json_schema()
    assert raw["title"] == "m"  # pydantic adds this label
    cleaned = ArgsModelBuilder.clean_schema(raw)
    assert "title" not in cleaned  # label gone
    assert "title" in cleaned["properties"]  # the FIELD survives
    assert "title" not in cleaned["properties"]["title"]


def test_annotated_description_is_kept():
    def f(city: Annotated[str, Field(description="City name")]) -> None:
        """Doc."""

    assert schema_of(f)["properties"]["city"]["description"] == "City name"


def test_docstring_args_become_descriptions():
    def f(a: int, b: int = 2) -> int:
        """Add.

        Args:
            a: first number
            b: second number
        """

    _, doc_params = parse_docstring(f.__doc__)
    props = schema_of(f, doc_params)["properties"]
    assert props["a"]["description"] == "first number"
    assert props["b"]["description"] == "second number"
    assert schema_of(f, doc_params)["required"] == ["a"]  # still required


def test_annotated_description_beats_docstring():
    def f(city: Annotated[str, Field(description="from Annotated")]) -> None:
        """Doc.

        Args:
            city: from docstring
        """

    _, doc_params = parse_docstring(f.__doc__)
    assert schema_of(f, doc_params)["properties"]["city"]["description"] == "from Annotated"


def test_parse_docstring_splits_summary_and_params():
    doc = """Save a note.

    Args:
        text: the note content,
            continued here
        tag (str): a label

    Returns:
        confirmation: this is not a parameter
    """
    summary, params = parse_docstring(doc)
    assert summary == "Save a note."
    assert params == {"text": "the note content, continued here", "tag": "a label"}


def test_parse_docstring_without_docstring():
    assert parse_docstring(None) == ("", {})


def test_format_errors_is_readable():
    model = ArgsModelBuilder(weather, "m").build()
    with pytest.raises(ValidationError) as info:
        model.model_validate({"cty": "x"})
    message = formate_errors(info.value)
    assert "city: Field required" in message
    assert "cty: Extra inputs are not permitted" in message
