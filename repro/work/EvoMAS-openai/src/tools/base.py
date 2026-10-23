"""
Base tool class for agent tools.
"""

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Base class for all agent tools."""

    def __init__(self, name: str, description: str):
        """Initialize base tool.

        Args:
            name: Name of the tool
            description: Description of what the tool does
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result
        """
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling.

        Returns:
            Tool schema dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema()
        }

    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool.

        Returns:
            Parameters schema dictionary
        """
        pass
