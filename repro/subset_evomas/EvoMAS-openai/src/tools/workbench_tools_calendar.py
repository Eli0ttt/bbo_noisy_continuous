"""
WorkBench Calendar Tools - Adapted for EVOMAS/smolagents.
Data source: dataset/workbench/calendar/data.csv

Note: Tool names use WorkBench format (calendar.tool_name) for compatibility.
"""

import pandas as pd
from smolagents import tool
from pathlib import Path
import json

DATA_PATH = Path(__file__).parent.parent.parent / "dataset" / "workbench" / "calendar" / "data.csv"
CALENDAR_EVENTS = pd.read_csv(DATA_PATH, dtype=str)
ORIGINAL_EVENTS = CALENDAR_EVENTS.copy()

def reset_state():
    global CALENDAR_EVENTS
    CALENDAR_EVENTS = ORIGINAL_EVENTS.copy()

@tool
def calendar_get_event_information_by_id(event_id: str = "", field: str = "") -> str:
    """
    Returns the event information for a given ID.

    Args:
        event_id (str): 8-digit ID of the event
        field (str): Field to return (event_id, event_name, participant_email, event_start, duration)

    Returns:
        str: Event information for the given ID and field
    """
    if not event_id:
        return "Event ID not provided."
    if not field:
        return "Field not provided."
    event = CALENDAR_EVENTS[CALENDAR_EVENTS["event_id"] == event_id].to_dict(orient="records")
    if event:
        if field in event[0]:
            return json.dumps({field: event[0][field]})
        else:
            return "Field not found."
    else:
        return "Event not found."

@tool
def calendar_search_events(query: str = "", time_min: str = "", time_max: str = "") -> str:
    """
    Returns events matching the given query.

    Args:
        query (str): Query to search for in event_name and participant_email fields
        time_min (str): Lower bound for event end time (YYYY-MM-DD HH:MM:SS)
        time_max (str): Upper bound for event start time (YYYY-MM-DD HH:MM:SS)

    Returns:
        str: JSON list of events matching the query (up to 5 results)
    """
    events = CALENDAR_EVENTS[
        (CALENDAR_EVENTS["event_name"].str.contains(query, case=False)) |
        (CALENDAR_EVENTS["participant_email"].str.contains(query, case=False))
    ].to_dict(orient="records")

    if time_min:
        events = [event for event in events if pd.Timestamp(event["event_start"]) >= pd.Timestamp(time_min)]
    if time_max:
        events = [event for event in events if pd.Timestamp(event["event_start"]) <= pd.Timestamp(time_max)]

    if events:
        return json.dumps(events[:5])
    else:
        return json.dumps([])

@tool
def calendar_create_event(event_name: str = "", participant_email: str = "", event_start: str = "", duration: str = "") -> str:
    """
    Creates a new calendar event.

    Args:
        event_name (str): Name of the event
        participant_email (str): Email of the participant
        event_start (str): Start time (YYYY-MM-DD HH:MM:SS)
        duration (str): Duration in minutes

    Returns:
        str: ID of the newly created event
    """
    global CALENDAR_EVENTS

    if not event_name:
        return "Event name not provided."
    if not participant_email:
        return "Participant email not provided."
    if not event_start:
        return "Event start not provided."
    if not duration:
        return "Event duration not provided."

    participant_email = participant_email.lower()
    event_id = str(int(CALENDAR_EVENTS["event_id"].max()) + 1).zfill(8)

    new_event = pd.DataFrame({
        "event_id": [event_id],
        "event_name": [event_name],
        "participant_email": [participant_email],
        "event_start": [event_start],
        "duration": [duration],
    })
    CALENDAR_EVENTS = pd.concat([CALENDAR_EVENTS, new_event], ignore_index=True)
    return event_id

@tool
def calendar_delete_event(event_id: str = "") -> str:
    """
    Deletes a calendar event.

    Args:
        event_id (str): 8-digit ID of the event

    Returns:
        str: Message indicating whether deletion was successful
    """
    global CALENDAR_EVENTS

    if not event_id:
        return "Event ID not provided."

    if event_id in CALENDAR_EVENTS["event_id"].values:
        CALENDAR_EVENTS = CALENDAR_EVENTS[CALENDAR_EVENTS["event_id"] != event_id]
        return "Event deleted successfully."
    else:
        return "Event not found."

@tool
def calendar_update_event(event_id: str = "", field: str = "", new_value: str = "") -> str:
    """
    Updates a calendar event.

    Args:
        event_id (str): 8-digit ID of the event
        field (str): Field to update
        new_value (str): New value for the field

    Returns:
        str: Message indicating whether update was successful
    """
    global CALENDAR_EVENTS

    if not event_id:
        return "Event ID not provided."
    if not field:
        return "Field not provided."
    if not new_value:
        return "New value not provided."

    if event_id in CALENDAR_EVENTS["event_id"].values:
        CALENDAR_EVENTS.loc[CALENDAR_EVENTS["event_id"] == event_id, field] = new_value
        return "Event updated successfully."
    else:
        return "Event not found."

def get_all_calendar_tools():
    return [
        calendar_get_event_information_by_id,
        calendar_search_events,
        calendar_create_event,
        calendar_delete_event,
        calendar_update_event
    ]
