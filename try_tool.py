from pydantic import BaseModel
from agent_tools import Tool

class Event(BaseModel):
    title : str 


def schedule(event : Event , hour : int = 9)-> str:
    """Schedule an event.

    Args:
        event: the event to schedule
        hour: hour of the day, 0-23
    """
    return f"{type(event).__name__}: {event.title} at {hour}:00"


t = Tool(schedule)
print(t.description)
print(t.to_openai())
print(t.to_openai()["function"]["parameters"]["properties"]["hour"])
print(t.invoke({"event": {"title": "Demo"}}))
