"""
WorkBench Tools - All Remaining Tools for EVOMAS/smolagents.
Includes: Analytics, Project Management, CRM, and Company Directory

Note: Tool names use WorkBench format (domain.tool_name) for compatibility.
"""

import pandas as pd
from smolagents import tool
from pathlib import Path
import json

# ============================================================================
# Analytics Tools  - /dataset/workbench/analytics/data.csv
# ============================================================================

ANALYTICS_PATH = Path(__file__).parent.parent.parent / "dataset" / "workbench" / "analytics" / "data.csv"
ANALYTICS_DATA = pd.read_csv(ANALYTICS_PATH, dtype=str)
ORIGINAL_ANALYTICS = ANALYTICS_DATA.copy()


def reset_analytics_state():
    """Resets the analytics data to the original state."""
    global ANALYTICS_DATA
    ANALYTICS_DATA = ORIGINAL_ANALYTICS.copy()


@tool
def analytics_get_visitor_information_by_id(visitor_id: str = "", field: str = "") -> str:
    """
    Get visitor information by ID.

    Args:
        visitor_id (str): Unique ID of the visitor
        field (str): Field to return (visitor_id, user_id, session_duration, traffic_source, device, date)

    Returns:
        str: Visitor information for the given ID and field
    """
    if not visitor_id:
        return "Visitor ID not provided."
    if not field:
        return "Field not provided."
    visitor = ANALYTICS_DATA[ANALYTICS_DATA["visitor_id"] == visitor_id].to_dict(orient="records")
    if visitor:
        if field in visitor[0]:
            return json.dumps({field: visitor[0][field]})
        else:
            return "Field not found."
    else:
        return "Visitor not found."

@tool
def analytics_total_visits_count(time_min: str = "", time_max: str = "") -> str:


    """
    Count total visits within a time range.

    Args:
        time_min (str): Lower date limit (YYYY-MM-DD)
        time_max (str): Upper date limit (YYYY-MM-DD)

    Returns:
        str: Total visit count
    """
    visits = ANALYTICS_DATA
    if time_min:
        visits = visits[pd.to_datetime(visits["date_of_visit"]).dt.date >= pd.to_datetime(time_min).date()]
    if time_max:
        visits = visits[pd.to_datetime(visits["date_of_visit"]).dt.date <= pd.to_datetime(time_max).date()]
    return str(len(visits))


@tool
def analytics_engaged_users_count(time_min: str = "", time_max: str = "") -> str:


    """
    Count engaged users (user_engaged==True) within time range.

    Args:
        time_min (str): Lower date limit (YYYY-MM-DD)
        time_max (str): Upper date limit (YYYY-MM-DD)

    Returns:
        str: Engaged user count
    """
    visits = ANALYTICS_DATA
    if time_min:
        visits = visits[pd.to_datetime(visits["date_of_visit"]).dt.date >= pd.to_datetime(time_min).date()]
    if time_max:
        visits = visits[pd.to_datetime(visits["date_of_visit"]).dt.date <= pd.to_datetime(time_max).date()]
    engaged = visits[visits["user_engaged"] == "True"]
    return str(len(engaged))


@tool
def analytics_traffic_source_count(time_min: str = "", time_max: str = "") -> str:


    """
    Count visits by traffic source within time range.

    Args:
        time_min (str): Lower date limit (YYYY-MM-DD)
        time_max (str): Upper date limit (YYYY-MM-DD)

    Returns:
        str: JSON dict with traffic source counts
    """
    visits = ANALYTICS_DATA
    if time_min:
        visits = visits[pd.to_datetime(visits["date_of_visit"]).dt.date >= pd.to_datetime(time_min).date()]
    if time_max:
        visits = visits[pd.to_datetime(visits["date_of_visit"]).dt.date <= pd.to_datetime(time_max).date()]
    counts = visits["traffic_source"].value_counts().to_dict()
    return json.dumps(counts)


@tool
def analytics_get_average_session_duration(time_min: str = "", time_max: str = "") -> str:


    """
    Get average session duration within time range.

    Args:
        time_min (str): Lower date limit (YYYY-MM-DD)
        time_max (str): Upper date limit (YYYY-MM-DD)

    Returns:
        str: Average session duration in seconds
    """
    visits = ANALYTICS_DATA
    if time_min:
        visits = visits[pd.to_datetime(visits["date_of_visit"]).dt.date >= pd.to_datetime(time_min).date()]
    if time_max:
        visits = visits[pd.to_datetime(visits["date_of_visit"]).dt.date <= pd.to_datetime(time_max).date()]
    avg = visits["session_duration_seconds"].astype(float).mean()
    return str(round(avg, 2))


# ============================================================================
# Project Management Tools - /dataset/workbench/project_management/data.csv
# ============================================================================

PM_PATH = Path(__file__).parent.parent.parent / "dataset" / "workbench" / "project_management" / "data.csv"
PROJECT_TASKS = pd.read_csv(PM_PATH, dtype=str)
ORIGINAL_TASKS = PROJECT_TASKS.copy()


def reset_project_management_state():
    """Resets the project management data to the original state."""
    global PROJECT_TASKS
    PROJECT_TASKS = ORIGINAL_TASKS.copy()


@tool
def project_management_get_task_information_by_id(task_id: str = "", field: str = "") -> str:


    """
    Get task information by ID.

    Args:
        task_id (str): Unique ID of the task
        field (str): Field to return (task_id, task_name, assignee_email, status, due_date, priority)

    Returns:
        str: Task information for the given ID and field
    """
    if not task_id:
        return "Task ID not provided."
    if not field:
        return "Field not provided."
    task = PROJECT_TASKS[PROJECT_TASKS["task_id"] == task_id].to_dict(orient="records")
    if task:
        if field in task[0]:
            return json.dumps({field: task[0][field]})
        else:
            return "Field not found."
    else:
        return "Task not found."


@tool
def project_management_search_tasks(task_name: str = "", assignee_email: str = "", status: str = "", priority: str = "") -> str:


    """
    Search for tasks by criteria.

    Args:
        task_name (str): Task name to search for
        assignee_email (str): Assignee email to filter by (matches assigned_to_email column)
        status (str): List name/status to filter by (e.g. "Backlog", "In Progress", "Completed")
        priority (str): Not used (for compatibility)

    Returns:
        str: JSON list of matching tasks (up to 5 results)
    """
    tasks = PROJECT_TASKS
    if task_name:
        tasks = tasks[tasks["task_name"].str.contains(task_name, case=False, na=False)]
    if assignee_email:
        tasks = tasks[tasks["assigned_to_email"].str.contains(assignee_email, case=False, na=False)]
    if status:
        tasks = tasks[tasks["list_name"].str.lower() == status.lower()]

    results = tasks.to_dict(orient="records")
    return json.dumps(results[:5] if results else [])


@tool
def project_management_create_task(task_name: str = "", assignee_email: str = "", status: str = "", due_date: str = "", board: str = "") -> str:


    """
    Create a new task.

    Args:
        task_name (str): Name of the task
        assignee_email (str): Email of assignee
        status (str): List name/status (e.g. "Backlog", "In Progress", "Completed")
        due_date (str): Due date (YYYY-MM-DD)
        board (str): Board name (e.g. "Front end", "Back end")

    Returns:
        str: ID of newly created task
    """
    global PROJECT_TASKS

    if not task_name:
        return "Task name not provided."
    if not assignee_email:
        return "Assignee email not provided."

    task_id = str(int(PROJECT_TASKS["task_id"].max()) + 1).zfill(8)
    new_task = pd.DataFrame([{
        "task_id": task_id,
        "task_name": task_name,
        "assigned_to_email": assignee_email.lower(),
        "list_name": status or "Backlog",
        "due_date": due_date,
        "board": board or ""
    }])
    PROJECT_TASKS = pd.concat([PROJECT_TASKS, new_task], ignore_index=True)
    return task_id


@tool
def project_management_delete_task(task_id: str = "") -> str:


    """
    Delete a task.

    Args:
        task_id (str): ID of the task to delete

    Returns:
        str: Confirmation message
    """
    global PROJECT_TASKS
    
    if not task_id:
        return "Task ID not provided."
    
    if task_id in PROJECT_TASKS["task_id"].values:
        PROJECT_TASKS = PROJECT_TASKS[PROJECT_TASKS["task_id"] != task_id]
        return "Task deleted successfully."
    else:
        return "Task not found."


@tool
def project_management_update_task(task_id: str = "", field: str = "", new_value: str = "") -> str:


    """
    Update a task.

    Args:
        task_id (str): ID of the task to update
        field (str): Field to update
        new_value (str): New value for the field

    Returns:
        str: Confirmation message
    """
    global PROJECT_TASKS
    
    if not task_id:
        return "Task ID not provided."
    if not field:
        return "Field not provided."
    if not new_value:
        return "New value not provided."
    
    if task_id in PROJECT_TASKS["task_id"].values:
        PROJECT_TASKS.loc[PROJECT_TASKS["task_id"] == task_id, field] = new_value
        return "Task updated successfully."
    else:
        return "Task not found."


# ============================================================================
# CRM Tools - /dataset/workbench/customer_relationship_manager/data.csv
# ============================================================================

CRM_PATH = Path(__file__).parent.parent.parent / "dataset" / "workbench" / "customer_relationship_manager" / "data.csv"
CRM_DATA = pd.read_csv(CRM_PATH, dtype=str)
ORIGINAL_CRM = CRM_DATA.copy()


def reset_crm_state():
    """Resets the CRM data to the original state."""
    global CRM_DATA
    CRM_DATA = ORIGINAL_CRM.copy()


@tool
def customer_relationship_manager_search_customers(customer_name: str = "", customer_email: str = "", status: str = "") -> str:


    """
    Search for customers by criteria.

    Args:
        customer_name (str): Customer name to search for
        customer_email (str): Customer email to filter by
        status (str): Status to filter by

    Returns:
        str: JSON list of matching customers (up to 5 results)
    """
    customers = CRM_DATA
    if customer_name:
        customers = customers[customers["customer_name"].str.contains(customer_name, case=False, na=False)]
    if customer_email:
        customers = customers[customers["customer_email"].str.contains(customer_email, case=False, na=False)]
    if status:
        customers = customers[customers["status"].str.lower() == status.lower()]
    
    results = customers.to_dict(orient="records")
    return json.dumps(results[:5] if results else [])


@tool
def customer_relationship_manager_add_customer(customer_name: str = "", customer_email: str = "", status: str = "") -> str:


    """
    Add a new customer.

    Args:
        customer_name (str): Name of the customer
        customer_email (str): Email of the customer
        status (str): Customer status

    Returns:
        str: ID of newly added customer
    """
    global CRM_DATA
    
    if not customer_name:
        return "Customer name not provided."
    if not customer_email:
        return "Customer email not provided."
    
    customer_id = str(int(CRM_DATA["customer_id"].max()) + 1).zfill(8)
    new_customer = pd.DataFrame([{
        "customer_id": customer_id,
        "customer_name": customer_name,
        "customer_email": customer_email.lower(),
        "status": status or "active"
    }])
    CRM_DATA = pd.concat([CRM_DATA, new_customer], ignore_index=True)
    return customer_id


@tool
def customer_relationship_manager_delete_customer(customer_id: str = "") -> str:


    """
    Delete a customer.

    Args:
        customer_id (str): ID of the customer to delete

    Returns:
        str: Confirmation message
    """
    global CRM_DATA
    
    if not customer_id:
        return "Customer ID not provided."
    
    if customer_id in CRM_DATA["customer_id"].values:
        CRM_DATA = CRM_DATA[CRM_DATA["customer_id"] != customer_id]
        return "Customer deleted successfully."
    else:
        return "Customer not found."


@tool
def customer_relationship_manager_update_customer(customer_id: str = "", field: str = "", new_value: str = "") -> str:


    """
    Update a customer.

    Args:
        customer_id (str): ID of the customer to update
        field (str): Field to update
        new_value (str): New value for the field

    Returns:
        str: Confirmation message
    """
    global CRM_DATA
    
    if not customer_id:
        return "Customer ID not provided."
    if not field:
        return "Field not provided."
    if not new_value:
        return "New value not provided."
    
    if customer_id in CRM_DATA["customer_id"].values:
        CRM_DATA.loc[CRM_DATA["customer_id"] == customer_id, field] = new_value
        return "Customer updated successfully."
    else:
        return "Customer not found."


# ============================================================================
# Company Directory Tool - Uses email data
# ============================================================================

EMAIL_PATH = Path(__file__).parent.parent.parent / "dataset" / "workbench" / "email" / "data.csv"
COMPANY_EMAILS = pd.read_csv(EMAIL_PATH, dtype=str)

@tool
def company_directory_find_email_address(name: str = "") -> str:


    """
    Find email address by name in company directory.

    Args:
        name (str): Name to search for

    Returns:
        str: Email address or error message
    """
    if not name:
        return "Name not provided."
    
    # Search in sender/recipient field
    matches = COMPANY_EMAILS[
        COMPANY_EMAILS["sender/recipient"].str.contains(name, case=False, na=False)
    ]["sender/recipient"].unique()
    
    if len(matches) > 0:
        return matches[0]
    else:
        return "Email address not found."


# ============================================================================
# Export Functions
# ============================================================================

def get_all_analytics_tools():
    return [
        analytics_get_visitor_information_by_id,
        analytics_total_visits_count,
        analytics_engaged_users_count,
        analytics_traffic_source_count,
        analytics_get_average_session_duration
    ]

def get_all_project_management_tools():
    return [
        project_management_get_task_information_by_id,
        project_management_search_tasks,
        project_management_create_task,
        project_management_delete_task,
        project_management_update_task
    ]

def get_all_crm_tools():
    return [
        customer_relationship_manager_search_customers,
        customer_relationship_manager_add_customer,
        customer_relationship_manager_delete_customer,
        customer_relationship_manager_update_customer
    ]

def get_all_company_directory_tools():
    return [company_directory_find_email_address]
