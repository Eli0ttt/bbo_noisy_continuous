"""
Prompts module for evomas.

This module provides prompt management, template rendering, and prompt registry functionality.
"""

from .registry import PromptRegistry
from .render import render_prompt

__all__ = ['PromptRegistry', 'render_prompt']
