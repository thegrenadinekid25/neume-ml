"""Audio I/O utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "configs/synthesis.yaml") -> Dict[str, Any]:
    """
    Load synthesis configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assumes this file is at src/utils/audio.py
    return Path(__file__).parent.parent.parent


def resolve_soundfont_path(config: Dict[str, Any]) -> str:
    """
    Resolve soundfont path from config, handling relative paths.

    Args:
        config: Loaded configuration dictionary

    Returns:
        Absolute path to soundfont file
    """
    sf_path = config["synthesis"]["soundfont"]["path"]

    # If already absolute, return as-is
    if Path(sf_path).is_absolute():
        return sf_path

    # Otherwise, resolve relative to project root
    return str(get_project_root() / sf_path)
