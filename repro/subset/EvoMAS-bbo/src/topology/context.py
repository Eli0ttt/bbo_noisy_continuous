"""
Context management for multi-agent systems.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class Context(BaseModel):
    """Context for multi-agent execution.

    Contains:
    - task: The original task/query
    - shared: Shared state accessible to all agents
    - reports: Agent outputs organized by agent_id
    - artifacts: Any artifacts produced during execution
    - trace: Execution trace for debugging
    """

    task: str = Field(..., description="The original task/query")
    shared: Dict[str, Any] = Field(default_factory=dict, description="Shared state")
    reports: Dict[str, str] = Field(default_factory=dict, description="Agent reports by agent_id")
    artifacts: Dict[str, Any] = Field(default_factory=dict, description="Execution artifacts")
    trace: List[Dict[str, Any]] = Field(default_factory=list, description="Execution trace")

    class Config:
        extra = "allow"

    def add_report(self, agent_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a report from an agent.

        Args:
            agent_id: ID of the agent
            content: Report content
            metadata: Optional metadata about the report
        """
        self.reports[agent_id] = content

        # Add to trace
        trace_entry = {
            "agent_id": agent_id,
            "action": "report",
            "content_length": len(content)
        }
        if metadata:
            trace_entry["metadata"] = metadata

        self.trace.append(trace_entry)

    def get_reports_for(self, agent_ids: List[str]) -> Dict[str, str]:
        """Get reports from specific agents.

        Args:
            agent_ids: List of agent IDs

        Returns:
            Dictionary of reports from specified agents
        """
        return {
            agent_id: self.reports[agent_id]
            for agent_id in agent_ids
            if agent_id in self.reports
        }

    def merge_context(self, strategy: str = "concat") -> str:
        """Merge all reports into a single context string.

        Args:
            strategy: Merge strategy ('concat', 'json', etc.)

        Returns:
            Merged context string
        """
        if strategy == "concat":
            parts = []
            for agent_id, report in self.reports.items():
                parts.append(f"=== {agent_id} ===\n{report}\n")
            return "\n".join(parts)
        elif strategy == "json":
            import json
            return json.dumps(self.reports, indent=2)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
