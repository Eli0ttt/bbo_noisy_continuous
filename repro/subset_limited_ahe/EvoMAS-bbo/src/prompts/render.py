"""
Prompt rendering utilities.
"""

from typing import Dict, Any


def render_prompt(template: str, variables: Dict[str, Any]) -> str:
    """Render a prompt template with variables.

    Uses simple {{variable}} syntax for variable substitution.

    Args:
        template: Template string with {{variable}} placeholders
        variables: Dictionary of variable values

    Returns:
        Rendered prompt string
    """
    result = template

    for key, value in variables.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, str(value))

    return result
