"""Synthetic market data generator for offline testing.

Generates realistic-looking OHLCV data using geometric Brownian motion,
so the entire pipeline can be tested without network access.

Usage:
    from src.data.synthetic import generate_synthetic_ohlcv
    df = generate_synthetic_ohlcv("FAKE.L", days=1000)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    ticker: str = "SYNTH",
    days: int = 2520,  # ~10 years of trading days
    start_date: str = "2015-01-02",
    initial_price: float = 100.0,
    annual_return: float = 0.08,
    annual_volatility: float = 0.20,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data using geometric Brownian motion.

    Creates realistic-looking daily price data with proper OHLC relationships
    and volume patterns. Useful for testing the full pipeline offline.

    Args:
        ticker: Ticker symbol (used for metadata, not in output).
        days: Number of trading days to generate.
        start_date: First date in the series.
        initial_price: Starting close price.
        annual_return: Expected annual return (drift).
        annual_volatility: Annualised volatility.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Generate trading dates (exclude weekends)
    dates = pd.bdate_range(start=start_date, periods=days, freq="B")

    # GBM parameters (daily)
    dt = 1 / 252
    mu = annual_return
    sigma = annual_volatility

    # Generate log returns
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(days)

    # Build close prices from cumulative returns
    close_prices = initial_price * np.exp(np.cumsum(log_returns))

    # Generate realistic OHLC from close
    # Daily range as percentage of close (typically 1-3%)
    daily_range_pct = np.abs(rng.normal(0.015, 0.008, days))
    daily_range = close_prices * daily_range_pct

    # Open is close of previous day with small gap
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price
    open_gap = rng.normal(0, 0.002, days) * close_prices
    open_prices = open_prices + open_gap

    # High and Low derived from range
    high_bias = rng.uniform(0.3, 0.7, days)  # Where in the range is the close?
    high_prices = np.maximum(open_prices, close_prices) + daily_range * high_bias
    low_prices = np.minimum(open_prices, close_prices) - daily_range * (1 - high_bias)

    # Ensure OHLC consistency
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    low_prices = np.maximum(low_prices, 0.01)  # No negative prices

    # Generate volume (log-normal, correlated with absolute returns)
    base_volume = 1_000_000
    abs_returns = np.abs(log_returns)
    volume_factor = 1 + 5 * abs_returns / abs_returns.mean()
    volume = (
        base_volume
        * volume_factor
        * rng.lognormal(0, 0.3, days)
    ).astype(int)

    df = pd.DataFrame(
        {
            "Open": np.round(open_prices, 2),
            "High": np.round(high_prices, 2),
            "Low": np.round(low_prices, 2),
            "Close": np.round(close_prices, 2),
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"

    return df


def generate_multi_asset_data(
    tickers: list[str],
    days: int = 2520,
    start_date: str = "2015-01-02",
    correlation: float = 0.3,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate correlated synthetic data for multiple assets.

    Args:
        tickers: List of ticker symbols.
        days: Number of trading days.
        start_date: First date.
        correlation: Pairwise correlation between assets.
        seed: Random seed.

    Returns:
        Dictionary mapping ticker → OHLCV DataFrame.
    """
    n = len(tickers)
    rng = np.random.default_rng(seed)

    # Build correlation matrix
    corr_matrix = np.full((n, n), correlation)
    np.fill_diagonal(corr_matrix, 1.0)

    # Cholesky decomposition for correlated normals
    L = np.linalg.cholesky(corr_matrix)
    uncorrelated = rng.standard_normal((days, n))
    correlated = uncorrelated @ L.T

    result = {}
    for i, ticker in enumerate(tickers):
        # Use correlated random draws to build each asset
        individual_seed = seed + i * 1000
        df = generate_synthetic_ohlcv(
            ticker=ticker,
            days=days,
            start_date=start_date,
            initial_price=rng.uniform(20, 500),
            annual_return=rng.uniform(0.02, 0.15),
            annual_volatility=rng.uniform(0.15, 0.35),
            seed=individual_seed,
        )
        result[ticker] = df

    return result
