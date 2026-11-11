"""
MAS (Multi-Agent System) module for evomas.

This module provides MAS specification, loading, runtime execution, and interpretation.
"""

from .spec import MasSpec
from .loader import load_mas_from_file
from .runtime import MasRuntime
from .interpreter import interpret_mas, run_mas_cli

__all__ = ['MasSpec', 'load_mas_from_file', 'MasRuntime', 'interpret_mas', 'run_mas_cli']
