"""Configuration loader for smart-dca.

Loads settings.yaml and assets.yaml, with environment variable overlay support.
Environment variables prefixed with SMARTDCA_ override YAML values using
double-underscore as path separator (e.g. SMARTDCA_SCORING__BUY_THRESHOLD=70).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.utils.exceptions import ConfigError

_settings_cache: dict | None = None
_assets_cache: dict | None = None

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_SETTINGS_PATH = _BASE_DIR / "config" / "settings.yaml"
_ASSETS_PATH = _BASE_DIR / "config" / "assets.yaml"

_REQUIRED_SETTINGS_KEYS = ["binance", "schedule", "scoring", "signals", "risk"]
_REQUIRED_ASSET_KEYS = ["weekly_allocation_gbp", "min_order_gbp", "enabled"]


def _coerce_value(value: str) -> int | float | bool | str:
    """Attempt to coerce a string env var to the appropriate Python type."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_env_overrides(settings: dict[str, Any]) -> dict[str, Any]:
    """Overlay environment variables prefixed with SMARTDCA_ onto settings.

    SMARTDCA_SCORING__BUY_THRESHOLD=70 sets settings["scoring"]["buy_threshold"] = 70.
    """
    prefix = "SMARTDCA_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path_parts = key[len(prefix) :].lower().split("__")
        target = settings
        for part in path_parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[path_parts[-1]] = _coerce_value(value)
    return settings


def load_settings(path: Path | None = None) -> dict:
    """Load settings from YAML file with environment variable overlays.

    Args:
        path: Optional path to settings.yaml. Defaults to config/settings.yaml.

    Returns:
        Settings dictionary.

    Raises:
        ConfigError: If file is missing or required keys are absent.
    """
    global _settings_cache
    if _settings_cache is not None:
        return _settings_cache

    settings_path = path or _SETTINGS_PATH
    if not settings_path.exists():
        raise ConfigError(f"Settings file not found: {settings_path}")

    with open(settings_path) as f:
        settings = yaml.safe_load(f)

    if settings is None:
        raise ConfigError(f"Settings file is empty: {settings_path}")

    for key in _REQUIRED_SETTINGS_KEYS:
        if key not in settings:
            raise ConfigError(f"Required settings section missing: '{key}'")

    settings = _apply_env_overrides(settings)
    _settings_cache = settings
    return settings


def load_assets(path: Path | None = None) -> dict:
    """Load asset configuration from YAML file.

    Args:
        path: Optional path to assets.yaml. Defaults to config/assets.yaml.

    Returns:
        Assets dictionary.

    Raises:
        ConfigError: If file is missing or required fields are absent.
    """
    global _assets_cache
    if _assets_cache is not None:
        return _assets_cache

    assets_path = path or _ASSETS_PATH
    if not assets_path.exists():
        raise ConfigError(f"Assets file not found: {assets_path}")

    with open(assets_path) as f:
        data = yaml.safe_load(f)

    if data is None or "assets" not in data:
        raise ConfigError(f"Assets file must contain 'assets' key: {assets_path}")

    for symbol, cfg in data["assets"].items():
        for key in _REQUIRED_ASSET_KEYS:
            if key not in cfg:
                raise ConfigError(f"Asset '{symbol}' missing required field: '{key}'")

    _assets_cache = data
    return data


def clear_cache() -> None:
    """Clear cached settings and assets. Used in tests."""
    global _settings_cache, _assets_cache
    _settings_cache = None
    _assets_cache = None
