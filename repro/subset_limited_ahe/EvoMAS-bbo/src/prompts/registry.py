"""
Prompt registry for managing prompt templates.
"""

import os
from pathlib import Path
from typing import Dict, Optional


class PromptRegistry:
    """Registry for managing prompt templates."""

    def __init__(self, templates_dir: Optional[str] = None):
        """Initialize prompt registry.

        Args:
            templates_dir: Directory containing prompt templates.
                         Defaults to src/prompts/templates/
        """
        if templates_dir is None:
            # Default to templates directory next to this file
            templates_dir = Path(__file__).parent / "templates"

        self.templates_dir = Path(templates_dir)
        self._templates: Dict[str, str] = {}

        # Load templates if directory exists
        if self.templates_dir.exists():
            self._load_templates()

    def _load_templates(self):
        """Load all prompt templates from the templates directory."""
        if not self.templates_dir.exists():
            return

        for template_file in self.templates_dir.glob("*.md"):
            template_name = template_file.stem
            with open(template_file, 'r', encoding='utf-8') as f:
                self._templates[template_name] = f.read()

    def get(self, template_name: str) -> Optional[str]:
        """Get a prompt template by name.

        Args:
            template_name: Name of the template (without .md extension)

        Returns:
            Template content or None if not found
        """
        return self._templates.get(template_name)

    def register(self, template_name: str, template_content: str):
        """Register a new prompt template.

        Args:
            template_name: Name for the template
            template_content: Content of the template
        """
        self._templates[template_name] = template_content

    def list_templates(self):
        """List all available template names."""
        return list(self._templates.keys())
