"""
Agent specification classes.
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


# Mapping from agent_type to backend runtime
# This allows auto-inferring the backend from agent_type
AGENT_TYPE_TO_BACKEND = {
    # smolagents agent types
    "CodeAgent": "smolagents",
    "ToolCallingAgent": "smolagents",
    # minisweagent agent types
    "DefaultAgent": "minisweagent",
    # sweagent agent types
    "SWEAgent": "sweagent",
}


def get_backend_for_agent_type(agent_type: str) -> Optional[str]:
    """Get the backend for a given agent type.

    Args:
        agent_type: The agent type (e.g., 'CodeAgent', 'DefaultAgent')

    Returns:
        The backend name or None if not found
    """
    return AGENT_TYPE_TO_BACKEND.get(agent_type)


class AgentSpec(BaseModel):
    """Specification for an agent (NOT implementation).

    This defines what an agent needs, not how it runs.
    """
    id: str = Field(..., description="Unique identifier for the agent")
    role: str = Field(..., description="Agent role (e.g., 'worker', 'aggregator', 'coordinator')")
    agent_type: str = Field(default="CodeAgent", description="Type of agent (CodeAgent, ToolCallingAgent, etc.)")
    model_id: str = Field(..., description="Model identifier (e.g., 'bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0')")
    prompt: Optional[str] = Field(default=None, description="Prompt template name or inline prompt")
    tools: List[str] = Field(default_factory=list, description="List of tool names available to this agent")
    max_tokens: int = Field(default=4096, description="Maximum tokens for model generation")
    temperature: float = Field(default=0.7, description="Temperature for model generation")
    device: Optional[str] = Field(default=None, description="Reserved for future use.")
    backend: Optional[str] = Field(default=None, description="Per-agent backend override (e.g., 'smolagents', 'minisweagent'). If None, uses MAS-level backend.")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional agent-specific parameters")

    class Config:
        extra = "allow"


class AgentResult(BaseModel):
    """Result from an agent execution."""
    agent_id: str = Field(..., description="ID of the agent that produced this result")
    content: str = Field(..., description="Generated content from the agent")
    success: bool = Field(default=True, description="Whether execution was successful")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about execution")

    class Config:
        extra = "allow"
