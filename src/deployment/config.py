"""Deployment-specific configuration loader.

Loads capital_deployment.yaml, validates required keys, and provides
convenience functions for tier selection and execution gating.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "capital_deployment.yaml"

# Required top-level sections in the deployment config.
_REQUIRED_SECTIONS = [
    "evaluation",
    "scanner",
    "sizing",
    "execution",
    "risk",
    "opus",
    "memory",
    "pipeline",
]


class DeploymentConfigError(Exception):
    """Raised when the deployment config is invalid or incomplete."""


def load_deployment_config(
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Load and validate the capital deployment configuration.

    Args:
        path: Path to YAML file. Defaults to config/capital_deployment.yaml.

    Returns:
        Fully-loaded config dict.

    Raises:
        FileNotFoundError: Config file missing.
        DeploymentConfigError: Required keys missing or malformed.
    """
    config_path = Path(path) if path else _CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Deployment config not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise DeploymentConfigError(
            "Deployment config must be a YAML mapping, "
            f"got {type(config).__name__}"
        )

    # Validate required sections exist.
    missing = [s for s in _REQUIRED_SECTIONS if s not in config]
    if missing:
        raise DeploymentConfigError(
            f"Deployment config missing required sections: {missing}"
        )

    # Validate evaluation weights sum to ~1.0.
    weights = config["evaluation"].get("weights", {})
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        raise DeploymentConfigError(
            f"Evaluation weights must sum to 1.0, got {total:.4f}"
        )

    return config


def get_active_tier(
    capital_gbp: float,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the sizing tier constraints for a given capital amount.

    Args:
        capital_gbp: Available capital in GBP.
        config: Deployment config dict. Loaded if not provided.

    Returns:
        Dict with tier constraints (max_positions, min_position_gbp,
        prefer_etfs, label).

    Raises:
        DeploymentConfigError: If no tier matches the capital amount.
    """
    if config is None:
        config = load_deployment_config()

    tiers = config["sizing"]["scaling"]

    for tier_name, tier in tiers.items():
        min_cap = tier.get("min_capital", 0)
        max_cap = tier.get("max_capital", float("inf"))
        if min_cap <= capital_gbp < max_cap:
            return {
                "tier_name": tier_name,
                "label": tier.get("label", tier_name),
                "max_positions": tier.get(
                    "max_positions",
                    config["sizing"]["constraints"]["max_positions"],
                ),
                "min_position_gbp": tier.get(
                    "min_position_gbp",
                    config["sizing"]["constraints"]["min_position_gbp"],
                ),
                "prefer_etfs": tier.get("prefer_etfs", False),
            }

    raise DeploymentConfigError(
        f"No sizing tier found for capital GBP {capital_gbp:.2f}"
    )


def get_structural_exclusions(
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Return the list of structurally excluded tickers from config.

    These are tickers that should never be scanned regardless of
    portfolio state (e.g. delisted, bad data, personal preference).

    Args:
        config: Deployment config dict. Loaded if not provided.

    Returns:
        List of ticker strings (usually empty).
    """
    if config is None:
        config = load_deployment_config()

    return list(
        config.get("scanner", {})
        .get("global_exclusions", {})
        .get("structural", [])
    )


def is_live_execution_enabled(
    config: dict[str, Any] | None = None,
) -> bool:
    """Check whether live execution is fully enabled in config.

    Both ``execution.mode`` must be ``"LIVE"`` and ``execution.confirm_live``
    must be ``true``. If either is missing or false, live execution is blocked.

    Args:
        config: Deployment config dict. Loaded if not provided.

    Returns:
        True only if both gates are satisfied.
    """
    if config is None:
        config = load_deployment_config()

    execution = config.get("execution", {})
    mode = str(execution.get("mode", "")).upper()
    confirm = execution.get("confirm_live", False)

    return mode == "LIVE" and bool(confirm)


def get_evaluation_weights(
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return the evaluation dimension weights.

    Args:
        config: Deployment config dict. Loaded if not provided.

    Returns:
        Dict mapping dimension name to weight (sums to 1.0).
    """
    if config is None:
        config = load_deployment_config()

    return dict(config["evaluation"]["weights"])


def get_opus_config(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Opus analyst configuration.

    Args:
        config: Deployment config dict. Loaded if not provided.

    Returns:
        Dict with model, max_tokens, timeout, max_retries, cost_warning_gbp.
    """
    if config is None:
        config = load_deployment_config()

    opus = dict(config["opus"])

    # Check for API key availability.
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    opus["api_key_available"] = bool(api_key)

    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set -- Opus analyst will use keyword fallback"
        )

    return opus
