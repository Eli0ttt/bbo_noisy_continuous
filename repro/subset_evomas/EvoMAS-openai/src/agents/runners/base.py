"""
Base agent runner interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..spec import AgentSpec, AgentResult


class BaseAgentRunner(ABC):
    """Base class for agent runners.

    Agent runners are responsible for:
    1. Converting AgentSpec into an executable agent instance
    2. Running the agent with context
    3. Returning standardized AgentResult
    """

    @abstractmethod
    def run(self, spec: AgentSpec, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Run an agent based on its specification.

        Args:
            spec: Agent specification
            task: Task/query for the agent to process
            context: Optional context from other agents or the environment

        Returns:
            AgentResult with execution outcome
        """
        pass

    @abstractmethod
    def create_agent(self, spec: AgentSpec) -> Any:
        """Create an agent instance from specification.

        Args:
            spec: Agent specification

        Returns:
            Framework-specific agent instance
        """
        pass
