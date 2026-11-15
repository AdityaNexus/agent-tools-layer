import time

import pytest

from agent_tools import ConfirmationHook, Hook, ToolCall, ToolRegistry


def names(registry):
    return [s["function"]["name"] for s in registry.schemas()]


@pytest.fixture
def registry():
    r = ToolRegistry()

    @r.register
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    return r


# ---------- registering ----------
def test_register_returns_the_function_unchanged(registry):
    @registry.register
    def mul(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    assert mul(3, 4) == 12  # still a normal function
    assert names(registry) == ["add", "mul"]


def test_duplicate_name_is_rejected(registry):
    def add(a: int) -> int:
        """Another add."""

    with pytest.raises(ValueError, match="already registered"):
        registry.register(add)


# ---------- executing ----------
def test_execute_success(registry):
    result = registry.execute(ToolCall("1", "add", {"a": 2, "b": 3}))
    assert result.ok and result.output == 5 and result.content == "5"


def test_unknown_tool_lists_available_tools(registry):
    result = registry.execute(ToolCall("1", "nope", "{}"))
    assert not result.ok
    assert "unknown tool" in result.error and "add" in result.error


def test_execute_all_never_raises(registry):
    calls = [
        ToolCall("1", "add", {"a": 2, "b": 3}),
        ToolCall("2", "add", {"a": 1}),
        ToolCall("3", "nope", "{}"),
    ]
    results = registry.execute_all(calls)
    assert [r.ok for r in results] == [True, False, False]
    assert "b: Field required" in results[1].error


def test_non_string_output_is_json_text():
    r = ToolRegistry()

    @r.register
    def data() -> dict:
        """Return data."""
        return {"a": 1}

    assert r.execute(ToolCall("1", "data", "{}")).content == '{"a": 1}'


def test_duration_is_measured_for_the_function_only():
    r = ToolRegistry()

    @r.register
    def slow() -> str:
        """Sleep a bit."""
        time.sleep(0.02)
        return "done"

    assert r.execute(ToolCall("1", "slow", "{}")).duration_ms >= 10
    assert (
        r.execute(ToolCall("2", "slow", '{"x": 1}')).duration_ms == 0.0
    )  # failed validation, never ran


# ---------- hooks ----------
class Recorder(Hook):
    def __init__(self, label, log):
        self.label, self.log = label, log

    def before_call(self, tool, kwargs):
        self.log.append(f"{self.label}:before:{tool.name}")
        return kwargs

    def after_call(self, tool, result):
        self.log.append(f"{self.label}:after:{result.ok}")
        return result


def test_hooks_run_in_order():
    log = []
    r = ToolRegistry(hooks=[Recorder("A", log), Recorder("B", log)])

    @r.register
    def ping() -> str:
        """Ping."""
        return "pong"

    r.execute(ToolCall("1", "ping", "{}"))
    assert log == ["A:before:ping", "B:before:ping", "A:after:True", "B:after:True"]


def test_after_hooks_also_see_failures_but_before_hooks_skip_invalid_calls():
    log = []
    r = ToolRegistry(hooks=[Recorder("A", log)])

    @r.register
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    r.execute(ToolCall("1", "add", {"a": 1}))  # fails validation
    assert log == ["A:after:False"]


def test_before_hook_can_change_arguments_and_after_hook_can_change_the_result():
    class Tweak(Hook):
        def before_call(self, tool, kwargs):
            return {**kwargs, "a": 10}

        def after_call(self, tool, result):
            result.output = f"<{result.output}>"
            return result

    r = ToolRegistry(hooks=[Tweak()])

    @r.register
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    assert r.execute(ToolCall("1", "add", {"a": 1, "b": 2})).output == "<12>"


# ---------- confirmation (human in the loop) ----------
def make_guarded(ask):
    ran = []
    r = ToolRegistry(hooks=[ConfirmationHook(ask=ask)])

    @r.register(requires_confirmation=True)
    def delete_file(path: str) -> str:
        """Delete a file."""
        ran.append(path)
        return f"deleted {path}"

    return r, ran


def test_approved_call_runs():
    r, ran = make_guarded(lambda tool, kwargs: True)
    assert r.execute(ToolCall("1", "delete_file", {"path": "/tmp/x"})).ok
    assert ran == ["/tmp/x"]


def test_denied_call_does_not_run_and_the_model_is_told():
    r, ran = make_guarded(lambda tool, kwargs: False)
    result = r.execute(ToolCall("1", "delete_file", {"path": "/tmp/x"}))
    assert not result.ok and "did not approve" in result.error
    assert ran == []


def test_the_approver_sees_validated_arguments():
    seen = []
    r, _ = make_guarded(lambda tool, kwargs: seen.append((tool.name, kwargs)) or True)
    r.execute(ToolCall("1", "delete_file", '{"path": "/tmp/x"}'))  # JSON string in...
    assert seen == [("delete_file", {"path": "/tmp/x"})]  # ...clean dict out


def test_tools_without_the_flag_are_never_asked_about():
    def must_not_ask(tool, kwargs):
        raise AssertionError("asked about a normal tool")

    r = ToolRegistry(hooks=[ConfirmationHook(ask=must_not_ask)])

    @r.register
    def ping() -> str:
        """Ping."""
        return "pong"

    assert r.execute(ToolCall("1", "ping", "{}")).ok


def test_fails_closed_when_no_confirmation_hook_is_configured():
    r = ToolRegistry()
    ran = []

    @r.register(requires_confirmation=True)
    def delete_file(path: str) -> str:
        """Delete a file."""
        ran.append(path)

    result = r.execute(ToolCall("1", "delete_file", {"path": "/tmp/x"}))
    assert not result.ok and "no ConfirmationHook" in result.error
    assert ran == []


# ---------- add_object (OOP style) ----------
class Notes:
    def __init__(self):
        self.items = []

    def save_note(self, text: str) -> str:
        """Save a short note.

        Args:
            text: the note content
        """
        self.items.append(text)
        return f"saved ({len(self.items)})"

    def list_notes(self) -> list:
        """List all notes."""
        return self.items

    def clear_notes(self) -> str:
        """Delete all notes."""
        self.items.clear()
        return "cleared"

    def no_docstring(self):
        return 1

    def _private(self):
        """Hidden."""


def test_add_object_registers_documented_public_methods_only():
    r = ToolRegistry()
    r.add_object(Notes(), prefix="notes_")
    assert names(r) == ["notes_clear_notes", "notes_list_notes", "notes_save_note"]


def test_add_object_methods_share_the_objects_state():
    r, notes = ToolRegistry(), Notes()
    r.add_object(notes)
    r.execute(ToolCall("1", "save_note", {"text": "buy milk"}))
    assert r.execute(ToolCall("2", "list_notes", "{}")).output == ["buy milk"]
    assert notes.items == ["buy milk"]


def test_add_object_include_is_an_allow_list():
    r = ToolRegistry()
    r.add_object(Notes(), include=["save_note"])
    assert names(r) == ["save_note"]


def test_add_object_confirm_marks_methods_as_needing_approval():
    r = ToolRegistry(hooks=[ConfirmationHook(ask=lambda tool, kwargs: False)])
    notes = Notes()
    r.add_object(notes, confirm=["clear_notes"])
    notes.items.append("keep me")
    assert not r.execute(ToolCall("1", "clear_notes", "{}")).ok  # denied
    assert notes.items == ["keep me"]
    assert r.execute(ToolCall("2", "list_notes", "{}")).ok  # other methods unaffected
