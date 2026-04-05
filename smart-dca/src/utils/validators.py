"""Input validation helpers for smart-dca."""

from __future__ import annotations

from src.utils.config_loader import load_assets, load_settings


def validate_asset(symbol: str) -> bool:
    """Check whether a symbol is a configured asset.

    Args:
        symbol: Trading pair symbol (e.g. 'BTCGBP').

    Returns:
        True if the symbol exists in assets.yaml, False otherwise.
    """
    assets = load_assets()
    return symbol in assets.get("assets", {})


def validate_amount_gbp(amount: float, asset: str) -> bool:
    """Validate that a GBP amount is within configured limits.

    Args:
        amount: Spend amount in GBP.
        asset: Trading pair symbol.

    Returns:
        True if amount is within min_order_gbp and max_single_order_gbp.
    """
    assets = load_assets()
    settings = load_settings()

    asset_cfg = assets.get("assets", {}).get(asset)
    if asset_cfg is None:
        return False

    min_order = asset_cfg.get("min_order_gbp", 0.0)
    max_order = settings.get("risk", {}).get("max_single_order_gbp", float("inf"))

    return min_order <= amount <= max_order


def validate_score(score: float) -> bool:
    """Validate that a composite score is within the 0-100 range.

    Args:
        score: Composite signal score.

    Returns:
        True if 0.0 <= score <= 100.0.
    """
    return 0.0 <= score <= 100.0
