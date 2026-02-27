"""Configuration loader — single source of truth for all settings.

Loads YAML config and resolves environment variable placeholders.
Usage:
    from src.utils.config import load_config
    cfg = load_config()
    tickers = cfg['universe']['tickers']
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${ENV_VAR} placeholders in config values."""
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.fullmatch(value)
        if match:
            return os.environ.get(match.group(1), "")
        return _ENV_VAR_PATTERN.sub(
            lambda m: os.environ.get(m.group(1), ""), value
        )
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and return the project configuration.

    Args:
        path: Path to YAML config file. Defaults to config/settings.yaml.

    Returns:
        Dictionary of resolved configuration values.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    return _resolve_env_vars(raw_config)


def get_data_dir(config: dict[str, Any] | None = None) -> Path:
    """Return the project data directory path."""
    if config is None:
        config = load_config()
    base = Path(__file__).parent.parent.parent
    return base / config["general"]["data_dir"]


def get_universe(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the active universe configuration."""
    if config is None:
        config = load_config()
    return config["universe"]
