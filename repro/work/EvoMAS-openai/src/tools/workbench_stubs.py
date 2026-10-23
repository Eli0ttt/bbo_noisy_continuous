"""
WorkBench tool stubs for CodeAgent.

These are lightweight stubs that provide tool descriptions to CodeAgent
without actual execution. The agent uses these to know what tools exist
and how to call them. Actual evaluation is done by WorkBench's evaluator.
"""

from smolagents import tool

# ============================================================================
# Email Tools
# ============================================================================

@tool
def email_send_email(recipient: str = "", subject: str = "", body: str = "") -> str:
    """
    Sends an email to a specified recipient.

    Args:
        recipient (str): Email address of the recipient
        subject (str): Subject line of the email
        body (str): Body content of the email

    Returns:
        str: Confirmation message
    """
    return f"email.send_email.func(recipient='{recipient}', subject='{subject}', body='{body}')"

@tool
def email_delete_email(email_id: str = "") -> str:
    """
    Deletes an email by its ID.

    Args:
        email_id (str): Unique ID of the email to delete

    Returns:
        str: Confirmation message
    """
    return f"email.delete_email.func(email_id='{email_id}')"

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
    return f"email.forward_email.func(email_id='{email_id}', recipient='{recipient}')"

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
    return f"email.reply_email.func(email_id='{email_id}', body='{body}')"

@tool
def email_get_email_information_by_id(email_id: str = "", field: str = "") -> str:
    """
    Retrieves specific details of an email by its ID.

    Args:
        email_id (str): Unique ID of the email
        field (str): Specific field to return (email_id, sender, subject, sent_date, body, inbox/outbox)

    Returns:
        str: Email information for the specified field
    """
    return f"email.get_email_information_by_id.func(email_id='{email_id}', field='{field}')"

@tool
def email_search_emails(query: str = "", date_min: str = "", date_max: str = "") -> str:
    """
    Searches for emails matching the given query.

    Args:
        query (str): Search query for subject, body, or sender
        date_min (str): Lower date limit (YYYY-MM-DD)
        date_max (str): Upper date limit (YYYY-MM-DD)

    Returns:
        str: List of matching emails
    """
    return f"email.search_emails.func(query='{query}', date_min='{date_min}', date_max='{date_max}')"


def get_workbench_email_tools():
    """Get all WorkBench email tools (stubs for CodeAgent)."""
    return [
        email_send_email,
        email_delete_email,
        email_forward_email,
        email_reply_email,
        email_get_email_information_by_id,
        email_search_emails
    ]
