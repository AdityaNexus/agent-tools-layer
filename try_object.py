import json

from agent_tools import ConfirmationHook, ToolCall, ToolRegistry


class Notes:
    def __init__(self):
        self.items = []

    def save_note(self, text: str) -> str:
        """Save a short note.

        Args:
            text: the note content
        """
        self.items.append(text)
        return f"saved ({len(self.items)} notes)"

    def list_notes(self) -> list[str]:
        """List all saved notes."""
        return self.items

    def clear_notes(self) -> str:
        """Delete all notes."""
        self.items.clear()
        return "cleared"

    def _internal(self) -> None:
        """Not exposed: starts with an underscore."""


notes = Notes()
registry = ToolRegistry(hooks=[ConfirmationHook(ask=lambda tool, kwargs: False)])  # always say no
print(registry.add_object(notes, prefix="notes_", confirm=["clear_notes"]))

print(json.dumps(registry.schemas()[2], indent=2))  # which tool is this? check the name

for call in [
    ToolCall("1", "notes_save_note", {"text": "buy milk"}),
    ToolCall("2", "notes_list_notes", "{}"),
    ToolCall("3", "notes_clear_notes", "{}"),
    ToolCall("4", "notes_list_notes", "{}"),
]:
    r = registry.execute(call)
    print(r.call_id, r.ok, r.content)
