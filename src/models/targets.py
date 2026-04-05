"""Target variable construction for ML models.

Provides functions to build prediction targets from price data.

**WARNING**: These functions use FUTURE prices to construct targets.
They must ONLY be used to create the *y* (target) variable, NEVER
as features.  Using forward-looking data as a feature constitutes
lookahead bias and will produce unrealistic backtest results.

Usage:
    from src.models.targets import create_direction_target, align_features_and_target
    target = create_direction_target(prices, horizon=5)
    X_aligned, y_aligned = align_features_and_target(features, target)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_direction_target(
    prices: pd.Series,
    horizon: int = 5,
) -> pd.Series:
    """Create a binary direction target from price data.

    Returns 1 if the price is higher in *horizon* trading days, 0 if
    lower or unchanged.

    **WARNING**: This function uses FUTURE prices.  The result must ONLY
    be used as a target variable (*y*), never as a feature.  The last
    *horizon* rows will be NaN because there is no future data available.

    Args:
        prices: Series of prices with DatetimeIndex (typically ``Close``).
        horizon: Number of trading days to look ahead.

    Returns:
        Series of 0/1 values with NaN for the last *horizon* rows.
        Named ``direction_{horizon}d``.
    """
    future_prices = prices.shift(-horizon)
    direction = pd.Series(np.nan, index=prices.index, name=f"direction_{horizon}d")
    mask = future_prices.notna()
    direction[mask] = (future_prices[mask] > prices[mask]).astype(float)
    return direction


def create_return_target(
    prices: pd.Series,
    horizon: int = 5,
    log: bool = True,
) -> pd.Series:
    """Create a forward-return target from price data.

    **WARNING**: This function uses FUTURE prices.  The result must ONLY
    be used as a target variable (*y*), never as a feature.  The last
    *horizon* rows will be NaN because there is no future data available.

    Args:
        prices: Series of prices with DatetimeIndex (typically ``Close``).
        horizon: Number of trading days to look ahead.
        log: If ``True``, return log returns; otherwise simple percentage
            returns.

    Returns:
        Series of forward returns with NaN for the last *horizon* rows.
        Named ``fwd_ret_{horizon}d``.
    """
    future_prices = prices.shift(-horizon)
    if log:
        ret = np.log(future_prices / prices)
    else:
        ret = future_prices / prices - 1.0
    ret.name = f"fwd_ret_{horizon}d"
    return ret


def align_features_and_target(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align features and target on their DatetimeIndex.

    Performs an inner join, drops any rows where either the features or
    the target contain NaN, and returns the aligned pair ready for
    model training.

    Args:
        features: Feature DataFrame with DatetimeIndex.
        target: Target Series with DatetimeIndex.

    Returns:
        Tuple of ``(X, y)`` where both share the same DatetimeIndex
        and contain no NaN values.
    """
    # Inner join on index
    combined = features.join(target, how="inner")

    rows_before = len(combined)

    # Drop rows with NaN in any column
    combined = combined.dropna()

    target_name = target.name or "target"
    X = combined.drop(columns=[target_name])
    y = combined[target_name]

    rows_dropped = rows_before - len(combined)
    logger.info(
        "align_features_and_target: %d rows aligned, %d dropped (NaN), "
        "date range %s -> %s",
        len(X),
        rows_dropped,
        X.index.min().date() if len(X) > 0 else "N/A",
        X.index.max().date() if len(X) > 0 else "N/A",
    )

    return X, y
