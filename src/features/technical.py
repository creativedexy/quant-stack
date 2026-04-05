"""Technical indicator functions for the quant stack.

Two API styles are provided:

* **``compute_*``** — Return a DataFrame of *indicator columns only*.
  These are the primary API and are used by :class:`FeaturePipeline`.
* **``add_*``** — Append indicator columns to the input DataFrame and
  return the combined result.  Retained for backward compatibility.

Every function reads default parameters from ``config['features']['technical']``
(with fallback to ``config['indicators']``) but accepts explicit overrides.
All rolling calculations enforce ``min_periods=window`` to prevent
lookahead bias.

Usage:
    from src.features.technical import compute_sma, compute_all_technical
    sma_df = compute_sma(ohlcv_df, windows=[10, 20])
    all_features = compute_all_technical(ohlcv_df)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ===================================================================
# Config helpers
# ===================================================================

def _features_cfg() -> dict[str, Any]:
    """Return ``config['features']['technical']``."""
    return load_config().get("features", {}).get("technical", {})


def _get_indicator_cfg(name: str) -> dict:
    """Return the legacy ``config['indicators'][name]`` section."""
    cfg = load_config()
    return cfg.get("indicators", {}).get(name, {})


# ===================================================================
# compute_* — return indicator columns only
# ===================================================================

def compute_sma(
    df: pd.DataFrame,
    windows: Sequence[int] | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Compute Simple Moving Averages.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        windows: Look-back windows.  Defaults to config
            ``features.technical.sma_windows``.
        column: Source column (default ``"Close"``).

    Returns:
        DataFrame with columns ``sma_{w}`` for each window *w*.
    """
    if windows is None:
        windows = _features_cfg().get(
            "sma_windows",
            _get_indicator_cfg("sma").get("windows", [5, 10, 20, 50, 200]),
        )
    series = df[column]
    result = pd.DataFrame(index=df.index)
    for w in windows:
        result[f"sma_{w}"] = series.rolling(window=w, min_periods=w).mean()
    return result


def compute_ema(
    df: pd.DataFrame,
    windows: Sequence[int] | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Compute Exponential Moving Averages.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        windows: EMA span values.  Defaults to config
            ``features.technical.ema_windows``.
        column: Source column (default ``"Close"``).

    Returns:
        DataFrame with columns ``ema_{w}`` for each span *w*.
    """
    if windows is None:
        windows = _features_cfg().get(
            "ema_windows",
            _get_indicator_cfg("ema").get("windows", [12, 26]),
        )
    series = df[column]
    result = pd.DataFrame(index=df.index)
    for w in windows:
        result[f"ema_{w}"] = series.ewm(
            span=w, min_periods=w, adjust=False,
        ).mean()
    return result


def compute_rsi(
    df: pd.DataFrame,
    window: int | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Compute the Relative Strength Index.

    Uses Wilder's smoothing (exponential MA with ``alpha = 1 / window``).
    Output is bounded 0–100.  Flat prices (zero change) yield NaN because
    both average gain and average loss are zero.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        window: Look-back period.  Defaults to config
            ``features.technical.rsi_window``.
        column: Source column (default ``"Close"``).

    Returns:
        DataFrame with a single column ``rsi_{window}``.
    """
    if window is None:
        window = _features_cfg().get(
            "rsi_window",
            _get_indicator_cfg("rsi").get("window", 14),
        )
    delta = df[column].diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    avg_gain = gains.ewm(
        alpha=1.0 / window, min_periods=window, adjust=False,
    ).mean()
    avg_loss = losses.ewm(
        alpha=1.0 / window, min_periods=window, adjust=False,
    ).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)

    return pd.DataFrame({f"rsi_{window}": rsi}, index=df.index)


def compute_macd(
    df: pd.DataFrame,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Compute MACD line, signal line, and histogram.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        fast: Fast EMA span.  Default from config.
        slow: Slow EMA span.  Default from config.
        signal: Signal line EMA span.  Default from config.
        column: Source column (default ``"Close"``).

    Returns:
        DataFrame with columns ``macd_line``, ``macd_signal``,
        ``macd_histogram``.
    """
    cfg = _features_cfg()
    icfg = _get_indicator_cfg("macd")
    fast = fast if fast is not None else cfg.get("macd_fast", icfg.get("fast", 12))
    slow = slow if slow is not None else cfg.get("macd_slow", icfg.get("slow", 26))
    signal = signal if signal is not None else cfg.get("macd_signal", icfg.get("signal", 9))

    series = df[column]
    fast_ema = series.ewm(span=fast, min_periods=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(
        span=signal, min_periods=signal, adjust=False,
    ).mean()
    macd_histogram = macd_line - macd_signal

    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_histogram,
    }, index=df.index)


def compute_bollinger_bands(
    df: pd.DataFrame,
    window: int | None = None,
    num_std: float | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Compute Bollinger Bands with normalised bandwidth.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        window: Rolling window size.  Default from config.
        num_std: Width in standard deviations.  Default from config.
        column: Source column (default ``"Close"``).

    Returns:
        DataFrame with columns ``bb_upper``, ``bb_middle``, ``bb_lower``,
        ``bb_width``.
    """
    cfg = _features_cfg()
    icfg = _get_indicator_cfg("bollinger_bands")
    window = window if window is not None else cfg.get("bb_window", icfg.get("window", 20))
    num_std = num_std if num_std is not None else cfg.get("bb_num_std", icfg.get("num_std", 2.0))

    series = df[column]
    rolling = series.rolling(window=window, min_periods=window)
    middle = rolling.mean()
    rolling_std = rolling.std(ddof=0)
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    width = (upper - lower) / middle

    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "bb_width": width,
    }, index=df.index)


def compute_atr(
    df: pd.DataFrame,
    window: int | None = None,
) -> pd.DataFrame:
    """Compute the Average True Range.

    True Range = max(High - Low, |High - PrevClose|, |Low - PrevClose|).
    ATR is the Wilder-smoothed (EMA) True Range.

    Args:
        df: OHLCV DataFrame (must contain ``High``, ``Low``, ``Close``).
        window: Smoothing window.  Default from config.

    Returns:
        DataFrame with a single column ``atr_{window}``.
    """
    if window is None:
        cfg = _features_cfg()
        window = cfg.get(
            "atr_window",
            _get_indicator_cfg("atr").get("window", 14),
        )
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    return pd.DataFrame({f"atr_{window}": atr}, index=df.index)


def compute_returns(
    df: pd.DataFrame,
    windows: Sequence[int] | None = None,
    log_returns: bool | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Compute returns over multiple horizons.

    Args:
        df: OHLCV DataFrame or Series with DatetimeIndex.
        windows: Return horizons in trading days.  Defaults to config
            ``features.returns.windows``.
        log_returns: If ``True``, compute log returns; otherwise simple
            percentage returns.  Defaults to config
            ``features.returns.log_returns``.
        column: Source column when *df* is a DataFrame.

    Returns:
        DataFrame with columns ``ret_{w}d`` for each window *w*.
    """
    cfg = load_config().get("features", {}).get("returns", {})
    if windows is None:
        windows = cfg.get("windows", [1, 5, 10, 21, 63, 252])
    if log_returns is None:
        log_returns = cfg.get("log_returns", True)

    series = df[column] if isinstance(df, pd.DataFrame) else df
    result = pd.DataFrame(index=series.index)
    for w in windows:
        if log_returns:
            result[f"ret_{w}d"] = np.log(series / series.shift(w))
        else:
            result[f"ret_{w}d"] = series.pct_change(w)
    return result


def compute_volatility(
    df: pd.DataFrame,
    windows: Sequence[int] | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Compute annualised rolling volatility from log returns.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        windows: Rolling windows in trading days.  Defaults to config
            ``features.technical.volatility_windows``.
        column: Source column (default ``"Close"``).

    Returns:
        DataFrame with columns ``vol_{w}d`` for each window *w*.
    """
    if windows is None:
        windows = _features_cfg().get("volatility_windows", [21, 63])

    log_ret = np.log(df[column] / df[column].shift(1))
    result = pd.DataFrame(index=df.index)
    for w in windows:
        result[f"vol_{w}d"] = (
            log_ret.rolling(window=w, min_periods=w).std(ddof=1) * np.sqrt(252)
        )
    return result


# ===================================================================
# compute_all_technical — convenience aggregator
# ===================================================================

def compute_all_technical(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute all technical features and return a features-only DataFrame.

    Calls every ``compute_*`` function with parameters from *config*
    (or the default ``config/settings.yaml``).

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        config: Full project config dict.  If ``None`` the default
            settings are loaded.

    Returns:
        DataFrame containing **only** indicator and return columns —
        the original OHLCV columns are **not** included.
    """
    if config is not None:
        cfg = config.get("features", {}).get("technical", {})
        ret_cfg = config.get("features", {}).get("returns", {})
    else:
        cfg = _features_cfg()
        ret_cfg = load_config().get("features", {}).get("returns", {})

    parts: list[pd.DataFrame] = [
        compute_sma(df, windows=cfg.get("sma_windows")),
        compute_ema(df, windows=cfg.get("ema_windows")),
        compute_rsi(df, window=cfg.get("rsi_window")),
        compute_macd(
            df,
            fast=cfg.get("macd_fast"),
            slow=cfg.get("macd_slow"),
            signal=cfg.get("macd_signal"),
        ),
        compute_bollinger_bands(
            df,
            window=cfg.get("bb_window"),
            num_std=cfg.get("bb_num_std"),
        ),
        compute_atr(df, window=cfg.get("atr_window")),
        compute_returns(
            df,
            windows=ret_cfg.get("windows"),
            log_returns=ret_cfg.get("log_returns"),
        ),
        compute_volatility(df, windows=cfg.get("volatility_windows")),
    ]

    result = pd.concat(parts, axis=1)

    logger.info(
        "compute_all_technical: %d features, %d rows",
        result.shape[1], result.shape[0],
    )
    return result


# ===================================================================
# add_* wrappers — backward compatibility
# ===================================================================

def add_sma(
    df: pd.DataFrame,
    windows: Sequence[int] | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Add SMA columns to *df*.  See :func:`compute_sma`."""
    if windows is None:
        windows = _get_indicator_cfg("sma").get("windows", [5, 10, 20, 50, 200])
    return df.join(compute_sma(df, windows=windows, column=column))


def add_ema(
    df: pd.DataFrame,
    windows: Sequence[int] | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Add EMA columns to *df*.  See :func:`compute_ema`."""
    if windows is None:
        windows = _get_indicator_cfg("ema").get("windows", [12, 26])
    return df.join(compute_ema(df, windows=windows, column=column))


def add_rsi(
    df: pd.DataFrame,
    window: int | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Add RSI column to *df*.  See :func:`compute_rsi`."""
    if window is None:
        window = _get_indicator_cfg("rsi").get("window", 14)
    return df.join(compute_rsi(df, window=window, column=column))


def add_macd(
    df: pd.DataFrame,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Add MACD columns to *df*.  See :func:`compute_macd`.

    Note: the histogram column is named ``macd_hist`` (not
    ``macd_histogram``) for backward compatibility.
    """
    icfg = _get_indicator_cfg("macd")
    fast = fast if fast is not None else icfg.get("fast", 12)
    slow = slow if slow is not None else icfg.get("slow", 26)
    signal = signal if signal is not None else icfg.get("signal", 9)

    macd_df = compute_macd(df, fast=fast, slow=slow, signal=signal, column=column)
    macd_df = macd_df.rename(columns={"macd_histogram": "macd_hist"})
    return df.join(macd_df)


def add_bollinger_bands(
    df: pd.DataFrame,
    window: int | None = None,
    num_std: float | None = None,
    *,
    column: str = "Close",
) -> pd.DataFrame:
    """Add Bollinger Band columns to *df*.  See :func:`compute_bollinger_bands`.

    Note: returns ``bb_upper``, ``bb_middle``, ``bb_lower`` (no ``bb_width``)
    for backward compatibility.
    """
    icfg = _get_indicator_cfg("bollinger_bands")
    window = window if window is not None else icfg.get("window", 20)
    num_std = num_std if num_std is not None else icfg.get("num_std", 2.0)

    bb_df = compute_bollinger_bands(
        df, window=window, num_std=num_std, column=column,
    )
    bb_df = bb_df.drop(columns=["bb_width"])
    return df.join(bb_df)


def add_atr(
    df: pd.DataFrame,
    window: int | None = None,
) -> pd.DataFrame:
    """Add ATR column to *df*.  See :func:`compute_atr`."""
    if window is None:
        window = _get_indicator_cfg("atr").get("window", 14)
    return df.join(compute_atr(df, window=window))


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every indicator with default (config) parameters.

    Returns the input DataFrame with all indicator columns appended.
    Uses the legacy ``config['indicators']`` section for defaults.
    """
    out = add_sma(df)
    out = add_ema(out)
    out = add_rsi(out)
    out = add_macd(out)
    out = add_bollinger_bands(out)
    out = add_atr(out)
    return out
