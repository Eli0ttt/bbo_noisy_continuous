"""
Tool registry for managing available tools.
"""

from typing import Dict, List, Optional
from .base import BaseTool


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a new tool.

        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_all_schemas(self) -> List[Dict]:
        """Get schemas for all registered tools.

        Returns:
            List of tool schemas
        """
        return [tool.get_schema() for tool in self._tools.values()]
