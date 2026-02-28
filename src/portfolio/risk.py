"""Standalone portfolio risk metric functions.

All functions operate on plain NumPy / Pandas objects and do not depend on
riskfolio-lib, so they can be used freely across the pipeline.

Every function accepts returns as a ``pd.DataFrame`` or ``pd.Series`` with a
``DatetimeIndex``.

Usage::

    from src.portfolio.risk import sharpe_ratio, max_drawdown, risk_summary
    sr = sharpe_ratio(daily_returns)
    dd = max_drawdown(daily_returns)
    summary = risk_summary(daily_returns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# 1. Portfolio returns
# ------------------------------------------------------------------

def portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Compute weighted portfolio return series.

    Args:
        asset_returns: Daily asset returns (tickers as columns,
            DatetimeIndex).
        weights: Portfolio weights indexed by ticker.  Must sum to
            approximately 1.0 (tolerance 0.01).

    Returns:
        Daily portfolio return Series with the same DatetimeIndex.

    Raises:
        ValueError: If weights do not sum to ~1.0 or columns mismatch.
    """
    weight_sum = float(weights.sum())
    if abs(weight_sum - 1.0) > 0.01:
        raise ValueError(
            f"Weights must sum to ~1.0 (tolerance 0.01), got {weight_sum:.6f}"
        )

    # Align weights to DataFrame columns
    w = weights.reindex(asset_returns.columns)
    if w.isna().any():
        missing = list(w[w.isna()].index)
        raise ValueError(
            f"Weight tickers not found in asset_returns columns: {missing}"
        )

    port_ret = asset_returns.values @ w.values
    return pd.Series(port_ret, index=asset_returns.index, name="portfolio_returns")


# ------------------------------------------------------------------
# 2. Sharpe ratio
# ------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.045,
    periods: int = 252,
) -> float:
    """Compute the annualised Sharpe ratio.

    Args:
        returns: Daily return series with DatetimeIndex.
        risk_free_rate: Annualised risk-free rate.
        periods: Number of trading periods per year (252 for daily).

    Returns:
        Annualised Sharpe ratio as a float.
    """
    daily_rf = (1 + risk_free_rate) ** (1 / periods) - 1
    excess = returns - daily_rf
    if excess.std(ddof=1) == 0:
        return float("inf") if excess.mean() > 0 else float("-inf")
    return float((excess.mean() / excess.std(ddof=1)) * np.sqrt(periods))


# ------------------------------------------------------------------
# 3. Sortino ratio
# ------------------------------------------------------------------

def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.045,
    periods: int = 252,
) -> float:
    """Compute the annualised Sortino ratio.

    Uses only downside deviation (returns below the risk-free rate) as
    the denominator.

    Args:
        returns: Daily return series with DatetimeIndex.
        risk_free_rate: Annualised risk-free rate.
        periods: Number of trading periods per year.

    Returns:
        Annualised Sortino ratio as a float.
    """
    daily_rf = (1 + risk_free_rate) ** (1 / periods) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = np.sqrt((downside**2).mean())
    return float((excess.mean() / downside_std) * np.sqrt(periods))


# ------------------------------------------------------------------
# 4. Max drawdown
# ------------------------------------------------------------------

def max_drawdown(returns: pd.Series) -> dict:
    """Compute the maximum drawdown and associated dates.

    Builds a cumulative wealth curve from daily returns
    (starting at 1.0), then identifies the peak-to-trough drawdown.

    Args:
        returns: Daily return series with DatetimeIndex.

    Returns:
        Dictionary with keys:
        - ``max_drawdown``: Maximum drawdown as a negative float
          (e.g. -0.25 for a 25% drawdown).  0.0 if no drawdown.
        - ``peak_date``: Date of the peak before the drawdown.
        - ``trough_date``: Date of the trough.
        - ``recovery_date``: Date when the equity curve recovered to
          the peak level, or ``None`` if not yet recovered.
    """
    if len(returns) == 0:
        return {
            "max_drawdown": 0.0,
            "peak_date": None,
            "trough_date": None,
            "recovery_date": None,
        }

    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    drawdowns = (wealth - running_max) / running_max

    mdd = float(drawdowns.min())
    if mdd == 0.0:
        return {
            "max_drawdown": 0.0,
            "peak_date": None,
            "trough_date": None,
            "recovery_date": None,
        }

    trough_date = drawdowns.idxmin()

    # Peak is the running max at the trough date
    peak_value = running_max.loc[trough_date]
    # Find the date of the peak (last date before trough where wealth == peak_value)
    pre_trough = wealth.loc[:trough_date]
    peak_date = pre_trough[pre_trough == peak_value].index[-1]

    # Recovery: first date after trough where wealth >= peak_value
    post_trough = wealth.loc[trough_date:]
    recovered = post_trough[post_trough >= peak_value]
    if len(recovered) > 0 and recovered.index[0] != trough_date:
        recovery_date = recovered.index[0]
    elif len(recovered) > 1:
        recovery_date = recovered.index[1]
    else:
        recovery_date = None

    return {
        "max_drawdown": mdd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
    }


# ------------------------------------------------------------------
# 5. Value at Risk
# ------------------------------------------------------------------

def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Compute historical Value-at-Risk for a return series.

    VaR is reported as a *positive* number representing the loss
    threshold exceeded with probability ``1 - confidence``.

    Args:
        returns: Daily return series.
        confidence: Confidence level (e.g. 0.95 for 95% VaR).

    Returns:
        Scalar VaR value (positive = loss).
    """
    quantile = np.percentile(returns.dropna(), (1 - confidence) * 100)
    return -float(quantile)


# ------------------------------------------------------------------
# 6. Conditional VaR (Expected Shortfall)
# ------------------------------------------------------------------

def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Compute historical Conditional Value-at-Risk (Expected Shortfall).

    CVaR is the mean of all losses beyond the VaR threshold, reported
    as a positive number.

    Args:
        returns: Daily return series.
        confidence: Confidence level.

    Returns:
        Scalar CVaR value (positive = loss).
    """
    clean = returns.dropna().values
    cutoff = np.percentile(clean, (1 - confidence) * 100)
    tail = clean[clean <= cutoff]
    if len(tail) == 0:
        return value_at_risk(returns, confidence)
    return -float(tail.mean())


# ------------------------------------------------------------------
# 7. Rolling Sharpe
# ------------------------------------------------------------------

def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.045,
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


# ------------------------------------------------------------------
# 8. Correlation report
# ------------------------------------------------------------------

def correlation_report(
    returns: pd.DataFrame,
    threshold: float = 0.85,
) -> dict:
    """Compute the return correlation matrix and flag high correlations.

    Args:
        returns: Asset returns (tickers as columns).
        threshold: Absolute correlation above which a pair is flagged.

    Returns:
        Dictionary with keys:
        - ``correlation_matrix``: Full correlation DataFrame.
        - ``high_pairs``: List of ``(ticker_a, ticker_b, correlation)``
          tuples for pairs exceeding the threshold.
    """
    corr = returns.corr()

    high_pairs: list[tuple[str, str, float]] = []
    n = corr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                high_pairs.append((
                    str(corr.index[i]),
                    str(corr.columns[j]),
                    float(val),
                ))
                logger.warning(
                    "High correlation detected between %s and %s: %.3f",
                    corr.index[i],
                    corr.columns[j],
                    val,
                )

    return {
        "correlation_matrix": corr,
        "high_pairs": high_pairs,
    }


# ------------------------------------------------------------------
# 9. Risk summary
# ------------------------------------------------------------------

def risk_summary(
    returns: pd.Series,
    risk_free_rate: float = 0.045,
) -> dict:
    """One-call risk summary combining all key metrics.

    Args:
        returns: Daily return series with DatetimeIndex.
        risk_free_rate: Annualised risk-free rate.

    Returns:
        Flat dictionary with keys: ``sharpe``, ``sortino``,
        ``max_drawdown``, ``var_95``, ``cvar_95``,
        ``annualised_return``, ``annualised_volatility``,
        ``calmar_ratio``.
    """
    ann_return = float((1 + returns.mean()) ** 252 - 1)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(252))
    mdd_result = max_drawdown(returns)
    mdd_value = mdd_result["max_drawdown"]

    if mdd_value != 0.0:
        calmar = ann_return / abs(mdd_value)
    else:
        calmar = float("inf") if ann_return > 0 else 0.0

    return {
        "sharpe": sharpe_ratio(returns, risk_free_rate=risk_free_rate),
        "sortino": sortino_ratio(returns, risk_free_rate=risk_free_rate),
        "max_drawdown": mdd_value,
        "var_95": value_at_risk(returns, confidence=0.95),
        "cvar_95": conditional_var(returns, confidence=0.95),
        "annualised_return": ann_return,
        "annualised_volatility": ann_vol,
        "calmar_ratio": calmar,
    }
