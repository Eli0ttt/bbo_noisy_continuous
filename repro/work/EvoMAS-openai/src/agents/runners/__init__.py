"""
Agent runners for different frameworks.

Runners are imported lazily to avoid loading unused packages.
Use direct imports when needed:
    from src.agents.runners.sweagent import SWEAgentRunner
    from src.agents.runners.minisweagent import MinisweagentRunner
    from src.agents.runners.smolagents import SmolagentsRunner
"""

from .base import BaseAgentRunner

__all__ = ['BaseAgentRunner']
