"""Standalone portfolio risk metric functions.

All functions operate on plain NumPy / Pandas objects and do not depend on
riskfolio-lib, so they can be used freely across the pipeline.

Usage:
    from src.portfolio.risk import portfolio_var, portfolio_cvar, max_drawdown
    var_95 = portfolio_var(returns, weights, confidence=0.95)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def portfolio_var(
    returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Compute historical Value-at-Risk for a weighted portfolio.

    VaR is reported as a *positive* number representing the loss threshold
    that is exceeded with probability ``1 - confidence``.

    Args:
        returns: Asset returns (tickers as columns, DatetimeIndex).
        weights: Portfolio weights aligned with *returns* columns.
        confidence: Confidence level (e.g. 0.95 for 95 % VaR).

    Returns:
        Scalar VaR value (positive = loss).
    """
    port_returns = _portfolio_returns(returns, weights)
    quantile = np.percentile(port_returns, (1 - confidence) * 100)
    return -float(quantile)


def portfolio_cvar(
    returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Compute historical Conditional Value-at-Risk (Expected Shortfall).

    CVaR is the mean of all losses beyond the VaR threshold, reported as a
    positive number.

    Args:
        returns: Asset returns (tickers as columns, DatetimeIndex).
        weights: Portfolio weights aligned with *returns* columns.
        confidence: Confidence level.

    Returns:
        Scalar CVaR value (positive = loss).
    """
    port_returns = _portfolio_returns(returns, weights)
    cutoff = np.percentile(port_returns, (1 - confidence) * 100)
    tail = port_returns[port_returns <= cutoff]
    return -float(tail.mean())


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute the maximum drawdown of an equity curve.

    Args:
        equity_curve: Cumulative equity / NAV series with DatetimeIndex.

    Returns:
        Maximum drawdown as a positive fraction (e.g. 0.25 = 25 %).
    """
    cumulative_max = equity_curve.cummax()
    drawdowns = (equity_curve - cumulative_max) / cumulative_max
    return -float(drawdowns.min())


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Compute rolling annualised Sharpe ratio.

    Args:
        returns: Daily portfolio or asset return series.
        window: Rolling window in trading days.
        risk_free_rate: Annualised risk-free rate.

    Returns:
        Series of rolling Sharpe ratios (NaN until *window* observations).
    """
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = returns - daily_rf
    rolling_mean = excess.rolling(window=window, min_periods=window).mean()
    rolling_std = excess.rolling(window=window, min_periods=window).std(ddof=1)
    sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    sharpe.name = "rolling_sharpe"
    return sharpe


def correlation_matrix(
    returns: pd.DataFrame,
    flag_threshold: float = 0.85,
) -> pd.DataFrame:
    """Compute the return correlation matrix and warn about high correlations.

    Args:
        returns: Asset returns (tickers as columns).
        flag_threshold: Absolute correlation above which a warning is logged.

    Returns:
        Correlation matrix as a DataFrame.
    """
    corr = returns.corr()

    # Flag highly correlated pairs (upper triangle only to avoid duplicates)
    n = corr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            val = abs(corr.iloc[i, j])
            if val >= flag_threshold:
                logger.warning(
                    "High correlation detected between %s and %s: %.3f",
                    corr.index[i],
                    corr.columns[j],
                    corr.iloc[i, j],
                )

    return corr


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
) -> np.ndarray:
    """Compute weighted portfolio return series as a 1-D NumPy array."""
    w = np.asarray(weights, dtype=float)
    return returns.values @ w
