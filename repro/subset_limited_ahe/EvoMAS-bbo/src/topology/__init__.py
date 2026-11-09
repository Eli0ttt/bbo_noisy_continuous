"""
Topology module for evomas.

This module provides context passing and communication management for multi-agent systems.
"""

from .context import Context
from .routing import RoutingConfig
from .merge import merge_reports

__all__ = ['Context', 'RoutingConfig', 'merge_reports']
