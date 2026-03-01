"""Risk metric utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def volatility(
    returns: ArrayLike, *, annualise: bool = True, periods_per_year: int = 252
) -> float:
    """Compute the standard deviation of returns, optionally annualised."""
    returns = np.asarray(returns, dtype=np.float64)
    vol = float(np.std(returns, ddof=1))
    if annualise:
        vol *= np.sqrt(periods_per_year)
    return vol


def sharpe_ratio(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    *,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualised Sharpe ratio."""
    returns = np.asarray(returns, dtype=np.float64)
    excess = returns - risk_free_rate / periods_per_year
    mean_excess = float(np.mean(excess))
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    return (mean_excess / std) * np.sqrt(periods_per_year)


def max_drawdown(prices: ArrayLike) -> float:
    """Compute the maximum drawdown from a price series.

    Returns a non-positive float representing the largest peak-to-trough
    decline as a fraction.
    """
    prices = np.asarray(prices, dtype=np.float64)
    peak = np.maximum.accumulate(prices)
    drawdowns = (prices - peak) / peak
    return float(np.min(drawdowns))
