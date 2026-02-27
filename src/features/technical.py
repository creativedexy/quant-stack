"""DataFrame-based technical indicators with config-driven defaults.

Every public function accepts a pandas DataFrame with OHLCV columns
(``Open``, ``High``, ``Low``, ``Close``, ``Volume``) and a DatetimeIndex.
Rolling calculations always set ``min_periods`` equal to the window size
to prevent lookahead bias during warm-up.

Usage:
    from src.features.technical import add_sma, add_all_indicators
    df = add_sma(ohlcv_df, windows=[10, 20])
    df = add_all_indicators(ohlcv_df)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_indicator_cfg(name: str) -> dict:
    """Return the config section for a named indicator."""
    cfg = load_config()
    return cfg["indicators"][name]


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


def add_sma(
    df: pd.DataFrame,
    windows: Sequence[int] | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Add Simple Moving Average columns.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        windows: Look-back windows.  Defaults to config ``indicators.sma.windows``.
        column: Source column (default ``"Close"``).

    Returns:
        DataFrame with new columns ``sma_{w}`` for each window *w*.
    """
    if windows is None:
        windows = _get_indicator_cfg("sma")["windows"]
    out = df.copy()
    for w in windows:
        out[f"sma_{w}"] = out[column].rolling(window=w, min_periods=w).mean()
    return out


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


def add_ema(
    df: pd.DataFrame,
    windows: Sequence[int] | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Add Exponential Moving Average columns.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        windows: Span values.  Defaults to config ``indicators.ema.windows``.
        column: Source column (default ``"Close"``).

    Returns:
        DataFrame with new columns ``ema_{w}`` for each span *w*.
    """
    if windows is None:
        windows = _get_indicator_cfg("ema")["windows"]
    out = df.copy()
    for w in windows:
        out[f"ema_{w}"] = out[column].ewm(span=w, min_periods=w, adjust=False).mean()
    return out


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def add_rsi(
    df: pd.DataFrame,
    window: int | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Add Relative Strength Index column.

    Uses the Wilder-smoothed (exponential) moving average of gains and losses.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        window: Look-back period.  Defaults to config ``indicators.rsi.window``.
        column: Source column (default ``"Close"``).

    Returns:
        DataFrame with new column ``rsi_{window}``.
    """
    if window is None:
        window = _get_indicator_cfg("rsi")["window"]
    out = df.copy()
    delta = out[column].diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # Wilder smoothing: alpha = 1/window
    avg_gain = gains.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    out[f"rsi_{window}"] = 100.0 - 100.0 / (1.0 + rs)
    return out


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def add_macd(
    df: pd.DataFrame,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Add MACD line, signal line, and histogram columns.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        fast: Fast EMA span.  Default from config.
        slow: Slow EMA span.  Default from config.
        signal: Signal line EMA span.  Default from config.
        column: Source column (default ``"Close"``).

    Returns:
        DataFrame with columns ``macd_line``, ``macd_signal``, ``macd_hist``.
    """
    cfg = _get_indicator_cfg("macd")
    fast = fast if fast is not None else cfg["fast"]
    slow = slow if slow is not None else cfg["slow"]
    signal = signal if signal is not None else cfg["signal"]

    out = df.copy()
    fast_ema = out[column].ewm(span=fast, min_periods=fast, adjust=False).mean()
    slow_ema = out[column].ewm(span=slow, min_periods=slow, adjust=False).mean()
    out["macd_line"] = fast_ema - slow_ema
    out["macd_signal"] = (
        out["macd_line"].ewm(span=signal, min_periods=signal, adjust=False).mean()
    )
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]
    return out


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def add_bollinger_bands(
    df: pd.DataFrame,
    window: int | None = None,
    num_std: float | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Add Bollinger Band columns.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        window: Rolling window size.  Default from config.
        num_std: Width in standard deviations.  Default from config.
        column: Source column (default ``"Close"``).

    Returns:
        DataFrame with columns ``bb_upper``, ``bb_middle``, ``bb_lower``.
    """
    cfg = _get_indicator_cfg("bollinger_bands")
    window = window if window is not None else cfg["window"]
    num_std = num_std if num_std is not None else cfg["num_std"]

    out = df.copy()
    rolling = out[column].rolling(window=window, min_periods=window)
    out["bb_middle"] = rolling.mean()
    rolling_std = rolling.std(ddof=0)
    out["bb_upper"] = out["bb_middle"] + num_std * rolling_std
    out["bb_lower"] = out["bb_middle"] - num_std * rolling_std
    return out


# ---------------------------------------------------------------------------
# ATR (Average True Range)
# ---------------------------------------------------------------------------


def add_atr(
    df: pd.DataFrame,
    window: int | None = None,
) -> pd.DataFrame:
    """Add Average True Range column.

    Computes the true range (max of high-low, |high-prev_close|,
    |low-prev_close|) then smooths with a Wilder EMA.

    Args:
        df: OHLCV DataFrame (must contain ``High``, ``Low``, ``Close``).
        window: Smoothing window.  Default from config.

    Returns:
        DataFrame with new column ``atr_{window}``.
    """
    if window is None:
        window = _get_indicator_cfg("atr")["window"]
    out = df.copy()
    high = out["High"]
    low = out["Low"]
    prev_close = out["Close"].shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    out[f"atr_{window}"] = (
        tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    )
    return out


# ---------------------------------------------------------------------------
# Convenience: add all indicators at once
# ---------------------------------------------------------------------------


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every indicator with default (config) parameters.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.

    Returns:
        DataFrame with all indicator columns appended.
    """
    out = add_sma(df)
    out = add_ema(out)
    out = add_rsi(out)
    out = add_macd(out)
    out = add_bollinger_bands(out)
    out = add_atr(out)
    return out
