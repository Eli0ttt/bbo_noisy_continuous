"""
Agents module for evomas.

This module provides agent specifications and runner implementations.
"""

from .spec import AgentSpec, AgentResult
from .runners.base import BaseAgentRunner

__all__ = ['AgentSpec', 'AgentResult', 'BaseAgentRunner']
