"""
Routing configuration for agent communication.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class RoutingConfig(BaseModel):
    """Configuration for how agents communicate and pass context.

    Defines:
    - Which agents report to which other agents
    - The order of execution
    - Communication edges between agents
    """

    reports_to: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping of agent_id to list of agent_ids it reports to"
    )

    edges: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Alternative edge-based representation (from -> to)"
    )

    class Config:
        extra = "allow"

    def get_dependencies(self, agent_id: str) -> List[str]:
        """Get agents that the given agent depends on (agents that report to it).

        Args:
            agent_id: ID of the agent

        Returns:
            List of agent IDs that this agent depends on
        """
        dependencies = []
        for source_id, targets in self.reports_to.items():
            if agent_id in targets:
                dependencies.append(source_id)
        return dependencies

    def get_execution_order(self, agent_ids: List[str]) -> List[str]:
        """Determine execution order using topological sort.

        Args:
            agent_ids: List of all agent IDs

        Returns:
            List of agent IDs in execution order
        """
        # Build dependency graph
        in_degree = {agent_id: 0 for agent_id in agent_ids}

        for agent_id in agent_ids:
            deps = self.get_dependencies(agent_id)
            in_degree[agent_id] = len(deps)

        # Topological sort (Kahn's algorithm)
        queue = [agent_id for agent_id, degree in in_degree.items() if degree == 0]
        execution_order = []

        while queue:
            queue.sort()  # For deterministic ordering
            current = queue.pop(0)
            execution_order.append(current)

            # Update in-degrees
            if current in self.reports_to:
                for target in self.reports_to[current]:
                    if target in in_degree:
                        in_degree[target] -= 1
                        if in_degree[target] == 0:
                            queue.append(target)

        # If not all agents are in order, there may be a cycle
        if len(execution_order) != len(agent_ids):
            # Fallback: return original order
            return agent_ids

        return execution_order

    def get_execution_levels(self, agent_ids: List[str]) -> List[List[str]]:
        """Group agents into dependency levels for parallel execution.

        Uses a level-based Kahn's algorithm to group zero-in-degree nodes
        per wave, enabling parallel execution within each level.

        Args:
            agent_ids: List of all agent IDs

        Returns:
            List of levels, where each level is a list of agent IDs
            that can execute in parallel
        """
        in_degree = {aid: 0 for aid in agent_ids}
        for aid in agent_ids:
            in_degree[aid] = len(self.get_dependencies(aid))

        current = sorted([aid for aid, deg in in_degree.items() if deg == 0])
        levels = []
        while current:
            levels.append(current)
            nxt = []
            for node in current:
                for target in (self.reports_to.get(node) or []):
                    if target in in_degree:
                        in_degree[target] -= 1
                        if in_degree[target] == 0:
                            nxt.append(target)
            current = sorted(nxt)

        if sum(len(l) for l in levels) != len(agent_ids):
            return [agent_ids]  # cycle fallback
        return levels
