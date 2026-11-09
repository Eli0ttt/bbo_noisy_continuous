"""
MAS specification using Pydantic.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from src.agents.spec import AgentSpec
from src.topology.routing import RoutingConfig


class ExecutionConfig(BaseModel):
    """Execution configuration for MAS."""
    parallel_workers: bool = Field(default=True, description="Execute worker agents in parallel")
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds for execution")
    max_retries: int = Field(default=0, description="Maximum number of retries on failure")
    debate_rounds: Optional[int] = Field(default=None, description="Number of debate rounds for iterative refinement (None = no debate)")
    smoa_layers: Optional[int] = Field(default=None, description="Number of layers in SMoA (None = not SMoA)")
    smoa_processors_per_layer: Optional[int] = Field(default=None, description="Number of processors per layer in SMoA")
    smoa_top_k: Optional[int] = Field(default=None, description="Top-k responses to select in SMoA judge")
    peer_review_stages: Optional[int] = Field(default=None, description="Enable peer review process (3 stages: create, review, revise)")
    croto_teams: Optional[int] = Field(default=None, description="Number of teams in CROTO (Cross-Team Orchestration)")
    croto_phases: Optional[int] = Field(default=None, description="Number of key phases for cross-team aggregation")
    croto_partition_size: Optional[int] = Field(default=2, description="Solutions per group in hierarchical partitioning")
    agent_config: str = Field(default="default", description="Agent configuration: 'default', 'simple', 'swebench', or path")

    class Config:
        extra = "allow"


class MasSpec(BaseModel):
    """Specification for a Multi-Agent System.

    This is a Pydantic model that can be loaded from YAML/JSON configuration files.
    """

    name: str = Field(..., description="Name of the MAS")
    description: Optional[str] = Field(default=None, description="Description of what this MAS does")
    backend: str = Field(default="smolagents", description="Backend runner (smolagents, sweagent, etc.)")

    agents: Dict[str, AgentSpec] = Field(..., description="Agent specifications by agent_id")
    topology: RoutingConfig = Field(..., description="Communication topology and routing")
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig, description="Execution configuration")

    class Config:
        extra = "allow"

    def get_execution_order(self) -> List[str]:
        """Get the execution order for agents based on topology.

        Returns:
            List of agent IDs in execution order
        """
        agent_ids = list(self.agents.keys())
        return self.topology.get_execution_order(agent_ids)

    def get_execution_levels(self) -> List[List[str]]:
        """Get DAG execution levels for parallel execution.

        Returns:
            List of levels, where each level is a list of agent IDs
            that can execute in parallel
        """
        return self.topology.get_execution_levels(list(self.agents.keys()))

    def get_worker_agents(self) -> List[str]:
        """Get list of worker agent IDs.

        Returns:
            List of agent IDs with role='worker'
        """
        return [
            agent_id
            for agent_id, spec in self.agents.items()
            if spec.role == "worker"
        ]

    def get_aggregator_agents(self) -> List[str]:
        """Get list of aggregator/coordinator agent IDs.

        Returns:
            List of agent IDs with role='aggregator'
        """
        return [
            agent_id
            for agent_id, spec in self.agents.items()
            if spec.role == "aggregator"
        ]

    def get_debater_agents(self) -> List[str]:
        """Get list of debater agent IDs.

        Returns:
            List of agent IDs with role='debater'
        """
        return [
            agent_id
            for agent_id, spec in self.agents.items()
            if spec.role == "debater"
        ]

    def get_processor_agents(self, layer: Optional[int] = None) -> List[str]:
        """Get list of processor agent IDs.

        Args:
            layer: If specified, only return processors from that layer

        Returns:
            List of agent IDs with role='processor'
        """
        processors = [
            agent_id
            for agent_id, spec in self.agents.items()
            if spec.role == "processor"
        ]

        if layer is not None:
            # Filter by layer if metadata exists
            processors = [
                agent_id
                for agent_id in processors
                if self.agents[agent_id].metadata and
                   self.agents[agent_id].metadata.get("layer") == layer
            ]

        return processors

    def get_judge_agent(self) -> Optional[str]:
        """Get judge agent ID.

        Returns:
            Agent ID with role='judge' or None
        """
        judges = [
            agent_id
            for agent_id, spec in self.agents.items()
            if spec.role == "judge"
        ]
        return judges[0] if judges else None

    def get_moderator_agent(self) -> Optional[str]:
        """Get moderator agent ID.

        Returns:
            Agent ID with role='moderator' or None
        """
        moderators = [
            agent_id
            for agent_id, spec in self.agents.items()
            if spec.role == "moderator"
        ]
        return moderators[0] if moderators else None

    def get_reviewer_agents(self) -> List[str]:
        """Get list of reviewer agent IDs.

        Returns:
            List of agent IDs with role='reviewer'
        """
        return [
            agent_id
            for agent_id, spec in self.agents.items()
            if spec.role == "reviewer"
        ]

    def get_team_worker_agents(self) -> List[str]:
        """Get list of team worker agent IDs.

        Returns:
            List of agent IDs with role='team_worker'
        """
        return [
            agent_id
            for agent_id, spec in self.agents.items()
            if spec.role == "team_worker"
        ]
