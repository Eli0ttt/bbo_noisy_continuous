"""
Merge strategies for combining agent reports.
"""

from typing import Dict, List, Any
import json


def merge_reports(reports: Dict[str, str], strategy: str = "concat", separator: str = "\n\n") -> str:
    """Merge multiple agent reports into a single string.

    Args:
        reports: Dictionary of agent_id -> report_content
        strategy: Merge strategy ('concat', 'json', 'structured')
        separator: Separator for concat strategy

    Returns:
        Merged report string
    """
    if strategy == "concat":
        return separator.join(reports.values())

    elif strategy == "json":
        return json.dumps(reports, indent=2)

    elif strategy == "structured":
        parts = []
        for agent_id, report in reports.items():
            parts.append(f"=== Agent: {agent_id} ===\n{report}")
        return separator.join(parts)

    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")


def aggregate_reports(reports: List[str], aggregation_fn: str = "concat") -> str:
    """Aggregate a list of reports.

    Args:
        reports: List of report strings
        aggregation_fn: Aggregation function ('concat', 'first', 'last')

    Returns:
        Aggregated report
    """
    if aggregation_fn == "concat":
        return "\n\n".join(reports)
    elif aggregation_fn == "first":
        return reports[0] if reports else ""
    elif aggregation_fn == "last":
        return reports[-1] if reports else ""
    else:
        raise ValueError(f"Unknown aggregation function: {aggregation_fn}")
