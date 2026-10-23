"""
WorkBench Email Tools - Adapted for EVOMAS/smolagents.

These tools provide email management functionality for WorkBench tasks.
Data source: dataset/workbench/email/data.csv
"""

import pandas as pd
from smolagents import tool
from pathlib import Path
import json

# Load email data from dataset-specific location
DATA_PATH = Path(__file__).parent.parent.parent / "dataset" / "workbench" / "email" / "data.csv"
EMAILS = pd.read_csv(DATA_PATH, dtype=str)

# For WorkBench evaluation - need to maintain original data
ORIGINAL_EMAILS = EMAILS.copy()


def reset_state():
    """Resets the emails to the original state."""
    global EMAILS
    EMAILS = ORIGINAL_EMAILS.copy()


@tool
def email_get_email_information_by_id(email_id: str = "", field: str = "") -> str:
    """
    Retrieves specific details of an email by its ID.

    Args:
        email_id (str): Unique ID of the email
        field (str): Specific field to return (email_id, sender, subject, sent_date, body, inbox/outbox)

    Returns:
        str: Information of the specified email for the given ID and field
    """
    if not email_id:
        return "Email ID not provided."
    if not field:
        return "Field not provided."

    email = EMAILS[EMAILS["email_id"] == email_id].to_dict(orient="records")
    if email:
        if field in email[0]:
            return json.dumps({field: email[0][field]})
        else:
            return "Field not found."
    else:
        return "Email not found."


@tool
def email_search_emails(query: str = "", date_min: str = "", date_max: str = "") -> str:
    """
    Searches for emails matching the given query across subject, body, or sender fields.
    The function matches an email if all words in the query appear in any of these fields.

    Args:
        query (str): Search query, matching terms in subject, body, or sender fields
        date_min (str): Lower date limit for the email's sent date (inclusive). Format: YYYY-MM-DD
        date_max (str): Upper date limit for the email's sent date (inclusive). Format: YYYY-MM-DD

    Returns:
        str: JSON list of emails matching the query criteria (up to 5 results)
    """
    query_words = query.lower().split()

    # Filter function to check if all query words are in any of the specified fields
    def filter_emails(row):
        combined_fields = f"{row['subject']} {row['body']} {row['sender/recipient']}".lower()
        return all(word in combined_fields for word in query_words)

    # Apply filter function across all rows
    filtered_emails = EMAILS.apply(filter_emails, axis=1)
    emails = EMAILS[filtered_emails].sort_values("sent_datetime", ascending=False).to_dict(orient="records")

    if date_min:
        emails = [
            email for email in emails
            if pd.Timestamp(email["sent_datetime"]).date() >= pd.Timestamp(date_min).date()
        ]
    if date_max:
        emails = [
            email for email in emails
            if pd.Timestamp(email["sent_datetime"]).date() <= pd.Timestamp(date_max).date()
        ]

    if len(emails):
        return json.dumps(emails[:5])
    else:
        return json.dumps([])


@tool
def email_send_email(recipient: str = "", subject: str = "", body: str = "") -> str:
    """
    Sends an email to a specified recipient.

    Args:
        recipient (str): Email address of the recipient
        subject (str): Subject line of the email
        body (str): Body content of the email

    Returns:
        str: Confirmation message with email details
    """
    global EMAILS

    if not recipient:
        return "Recipient not provided."
    if not subject:
        return "Subject not provided."
    if not body:
        return "Body not provided."

    # Generate new email ID
    new_email_id = str(int(EMAILS["email_id"].max()) + 1).zfill(8)

    # Get current timestamp
    from src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME
    sent_datetime = HARDCODED_CURRENT_TIME.strftime("%Y-%m-%d %H:%M:%S")

    # Create new email entry
    new_email = pd.DataFrame([{
        "email_id": new_email_id,
        "inbox/outbox": "outbox",
        "subject": subject,
        "sender/recipient": recipient,
        "sent_datetime": sent_datetime,
        "body": body
    }])

    EMAILS = pd.concat([EMAILS, new_email], ignore_index=True)

    return json.dumps({
        "email_id": new_email_id,
        "recipient": recipient,
        "subject": subject,
        "sent_datetime": sent_datetime,
        "message": "Email sent successfully."
    })


@tool
def email_delete_email(email_id: str = "") -> str:
    """
    Deletes an email by its ID.

    Args:
        email_id (str): Unique ID of the email to delete

    Returns:
        str: Confirmation message
    """
    global EMAILS

    if not email_id:
        return "Email ID not provided."

    initial_count = len(EMAILS)
    EMAILS = EMAILS[EMAILS["email_id"] != email_id]

    if len(EMAILS) < initial_count:
        return json.dumps({"message": f"Email {email_id} deleted successfully."})
    else:
        return "Email not found."


@tool
def email_forward_email(email_id: str = "", recipient: str = "") -> str:
    """
    Forwards an email to a specified recipient.

    Args:
        email_id (str): Unique ID of the email to forward
        recipient (str): Email address to forward to

    Returns:
        str: Confirmation message
    """
    global EMAILS

    if not email_id:
        return "Email ID not provided."
    if not recipient:
        return "Recipient not provided."

    # Find the original email
    email = EMAILS[EMAILS["email_id"] == email_id].to_dict(orient="records")
    if not email:
        return "Email not found."

    original = email[0]

    # Generate new email ID
    new_email_id = str(int(EMAILS["email_id"].max()) + 1).zfill(8)

    # Get current timestamp
    from src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME
    sent_datetime = HARDCODED_CURRENT_TIME.strftime("%Y-%m-%d %H:%M:%S")

    # Create forwarded email
    new_email = pd.DataFrame([{
        "email_id": new_email_id,
        "inbox/outbox": "outbox",
        "subject": f"Fwd: {original['subject']}",
        "sender/recipient": recipient,
        "sent_datetime": sent_datetime,
        "body": f"---------- Forwarded message ---------\n{original['body']}"
    }])

    EMAILS = pd.concat([EMAILS, new_email], ignore_index=True)

    return json.dumps({
        "email_id": new_email_id,
        "recipient": recipient,
        "message": "Email forwarded successfully."
    })


@tool
def email_reply_email(email_id: str = "", body: str = "") -> str:
    """
    Replies to an email.

    Args:
        email_id (str): Unique ID of the email to reply to
        body (str): Reply message body

    Returns:
        str: Confirmation message
    """
    global EMAILS

    if not email_id:
        return "Email ID not provided."
    if not body:
        return "Reply body not provided."

    # Find the original email
    email = EMAILS[EMAILS["email_id"] == email_id].to_dict(orient="records")
    if not email:
        return "Email not found."

    original = email[0]

    # Generate new email ID
    new_email_id = str(int(EMAILS["email_id"].max()) + 1).zfill(8)

    # Get current timestamp
    from src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME
    sent_datetime = HARDCODED_CURRENT_TIME.strftime("%Y-%m-%d %H:%M:%S")

    # Create reply email
    new_email = pd.DataFrame([{
        "email_id": new_email_id,
        "inbox/outbox": "outbox",
        "subject": f"Re: {original['subject']}",
        "sender/recipient": original['sender/recipient'],
        "sent_datetime": sent_datetime,
        "body": body
    }])

    EMAILS = pd.concat([EMAILS, new_email], ignore_index=True)

    return json.dumps({
        "email_id": new_email_id,
        "message": "Reply sent successfully."
    })


def get_all_email_tools():
    """Get all email tools for WorkBench tasks."""
    return [
        email_get_email_information_by_id,
        email_search_emails,
        email_send_email,
        email_delete_email,
        email_forward_email,
        email_reply_email
    ]
