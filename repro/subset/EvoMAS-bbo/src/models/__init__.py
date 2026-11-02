"""
Models module for evomas.

This module provides model loading and management functionality for various LLM providers.
"""

from .model import get_model, BedrockSmolagentsModel

__all__ = ['get_model', 'BedrockSmolagentsModel']
