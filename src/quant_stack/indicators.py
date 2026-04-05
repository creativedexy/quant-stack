"""Technical analysis indicators."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sma(prices: ArrayLike, *, window: int) -> NDArray[np.float64]:
    """Compute the Simple Moving Average.

    Parameters
    ----------
    prices : Price series.
    window : Look-back window size (must be >= 1).

    Returns
    -------
    Array of length ``len(prices) - window + 1``.
    """
    prices = np.asarray(prices, dtype=np.float64)
    if window < 1:
        raise ValueError("window must be >= 1")
    if window > len(prices):
        raise ValueError("window must be <= length of prices")
    cumsum = np.cumsum(prices)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    return cumsum[window - 1 :] / window


def ema(prices: ArrayLike, *, span: int) -> NDArray[np.float64]:
    """Compute the Exponential Moving Average.

    Parameters
    ----------
    prices : Price series.
    span : Span for the EMA decay factor (alpha = 2 / (span + 1)).

    Returns
    -------
    Array of the same length as *prices*.
    """
    prices = np.asarray(prices, dtype=np.float64)
    if span < 1:
        raise ValueError("span must be >= 1")
    alpha = 2.0 / (span + 1)
    out = np.empty_like(prices)
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(prices: ArrayLike, *, window: int = 14) -> NDArray[np.float64]:
    """Compute the Relative Strength Index.

    Uses the smoothed (Wilder) moving average method.

    Parameters
    ----------
    prices : Price series.
    window : Look-back period (default 14).

    Returns
    -------
    Array of length ``len(prices) - window``.  Values range from 0 to 100.
    """
    prices = np.asarray(prices, dtype=np.float64)
    if window < 1:
        raise ValueError("window must be >= 1")
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])

    rsi_values = np.empty(len(deltas) - window + 1)
    if avg_loss == 0:
        rsi_values[0] = 100.0
    else:
        rsi_values[0] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(1, len(rsi_values)):
        idx = window - 1 + i
        avg_gain = (avg_gain * (window - 1) + gains[idx]) / window
        avg_loss = (avg_loss * (window - 1) + losses[idx]) / window
        if avg_loss == 0:
            rsi_values[i] = 100.0
        else:
            rsi_values[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return rsi_values


def macd(
    prices: ArrayLike,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute MACD (Moving Average Convergence Divergence).

    Parameters
    ----------
    prices : Price series.
    fast : Span for the fast EMA (default 12).
    slow : Span for the slow EMA (default 26).
    signal : Span for the signal line EMA (default 9).

    Returns
    -------
    (macd_line, signal_line, histogram) — all arrays of the same length as
    *prices*.
    """
    prices = np.asarray(prices, dtype=np.float64)
    fast_ema = ema(prices, span=fast)
    slow_ema = ema(prices, span=slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, span=signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    prices: ArrayLike, *, window: int = 20, num_std: float = 2.0
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Bollinger Bands.

    Parameters
    ----------
    prices : Price series.
    window : Look-back window for the SMA and standard deviation (default 20).
    num_std : Number of standard deviations for the upper/lower bands
        (default 2.0).

    Returns
    -------
    (upper, middle, lower) — each of length ``len(prices) - window + 1``.
    """
    prices = np.asarray(prices, dtype=np.float64)
    middle = sma(prices, window=window)
    # Rolling standard deviation (population) matching the SMA window
    rolling_std = np.array(
        [
            np.std(prices[i : i + window])
            for i in range(len(prices) - window + 1)
        ]
    )
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    return upper, middle, lower
