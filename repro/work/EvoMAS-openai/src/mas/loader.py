"""
Loader for MAS configurations from files.
"""

import yaml
import json
from pathlib import Path
from typing import Union

from .spec import MasSpec


def load_mas_from_file(config_path: Union[str, Path]) -> MasSpec:
    """Load MAS specification from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        MasSpec instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is unsupported
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {config_path.suffix}")

    # Parse into MasSpec
    mas_spec = MasSpec(**config_dict)

    return mas_spec


def load_mas_from_dict(config_dict: dict) -> MasSpec:
    """Load MAS specification from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        MasSpec instance
    """
    return MasSpec(**config_dict)
