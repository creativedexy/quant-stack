"""Visualisation helpers for quick inspection of technical features.

Provides four plotting functions that create publication-ready matplotlib
figures from OHLCV data and computed feature DataFrames.

Usage:
    from src.features.visualisation import plot_price_with_bollinger
    plot_price_with_bollinger(ohlcv_df, features_df, ticker="SHEL.L")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Apply a clean style if available
try:
    plt.style.use("seaborn-v0_8")
except OSError:
    try:
        plt.style.use("seaborn")
    except OSError:
        pass


def plot_price_with_bollinger(
    df: pd.DataFrame,
    features: pd.DataFrame,
    ticker: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot Close price with Bollinger Bands overlay.

    Args:
        df: OHLCV DataFrame with DatetimeIndex and a ``Close`` column.
        features: Feature DataFrame containing ``bb_upper``, ``bb_middle``,
            ``bb_lower``.
        ticker: Ticker name shown in the title.
        save_path: If provided, the figure is saved to this path.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df.index, df["Close"], label="Close", linewidth=1, color="steelblue")

    if "bb_middle" in features.columns:
        ax.plot(
            features.index, features["bb_middle"],
            label="BB Middle", linewidth=0.8, color="orange", linestyle="--",
        )
    if "bb_upper" in features.columns and "bb_lower" in features.columns:
        upper = features["bb_upper"]
        lower = features["bb_lower"]
        ax.fill_between(
            features.index, lower, upper,
            alpha=0.15, color="orange", label="Bollinger Band",
        )

    title = "Close Price with Bollinger Bands"
    if ticker:
        title = f"{ticker} — {title}"
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        _save(fig, save_path)
    return fig


def plot_rsi(
    df: pd.DataFrame,
    features: pd.DataFrame,
    ticker: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot RSI with overbought / oversold zones.

    Args:
        df: OHLCV DataFrame (used for the DatetimeIndex).
        features: Feature DataFrame containing an ``rsi_*`` column.
        ticker: Ticker name shown in the title.
        save_path: If provided, the figure is saved to this path.

    Returns:
        The matplotlib Figure object.
    """
    rsi_col = _find_column(features, "rsi_")
    if rsi_col is None:
        raise KeyError("No RSI column found in features DataFrame")

    fig, ax = plt.subplots(figsize=(12, 6))
    rsi = features[rsi_col]

    ax.plot(rsi.index, rsi, label=rsi_col, linewidth=0.8, color="purple")
    ax.axhline(y=70, color="red", linestyle="--", linewidth=0.7, label="Overbought (70)")
    ax.axhline(y=30, color="green", linestyle="--", linewidth=0.7, label="Oversold (30)")
    ax.fill_between(rsi.index, 70, 100, alpha=0.08, color="red")
    ax.fill_between(rsi.index, 0, 30, alpha=0.08, color="green")
    ax.set_ylim(0, 100)

    title = "Relative Strength Index"
    if ticker:
        title = f"{ticker} — {title}"
    ax.set_title(title)
    ax.set_ylabel("RSI")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        _save(fig, save_path)
    return fig


def plot_macd(
    df: pd.DataFrame,
    features: pd.DataFrame,
    ticker: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot MACD line, signal line, and histogram.

    Args:
        df: OHLCV DataFrame (used for the DatetimeIndex).
        features: Feature DataFrame containing ``macd_line``,
            ``macd_signal``, and ``macd_histogram`` (or ``macd_hist``).
        ticker: Ticker name shown in the title.
        save_path: If provided, the figure is saved to this path.

    Returns:
        The matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [1, 1]}, sharex=True,
    )

    # MACD line + signal
    ax1.plot(
        features.index, features["macd_line"],
        label="MACD Line", linewidth=0.9, color="steelblue",
    )
    ax1.plot(
        features.index, features["macd_signal"],
        label="Signal", linewidth=0.9, color="orange",
    )
    ax1.axhline(y=0, color="grey", linewidth=0.5, linestyle="--")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    title = "MACD"
    if ticker:
        title = f"{ticker} — {title}"
    ax1.set_title(title)

    # Histogram
    hist_col = "macd_histogram" if "macd_histogram" in features.columns else "macd_hist"
    hist = features[hist_col]
    colours = ["green" if v >= 0 else "red" for v in hist]
    ax2.bar(features.index, hist, color=colours, alpha=0.6, width=1.5)
    ax2.axhline(y=0, color="grey", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Histogram")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        _save(fig, save_path)
    return fig


def plot_feature_correlations(
    features: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a heatmap of feature-feature correlations.

    Args:
        features: Feature DataFrame (numeric columns only).
        save_path: If provided, the figure is saved to this path.

    Returns:
        The matplotlib Figure object.
    """
    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "seaborn is required for correlation heatmaps. "
            "Install with: pip install seaborn"
        ) from exc

    # Select only numeric columns and drop constant columns
    numeric = features.select_dtypes(include="number")
    corr = numeric.corr()

    n_feats = len(corr)
    fig_size = max(8, min(n_feats * 0.5 + 2, 20))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        corr, ax=ax,
        cmap="RdBu_r", center=0,
        annot=n_feats <= 25,
        fmt=".1f" if n_feats <= 25 else "",
        annot_kws={"size": 7},
        square=True,
    )
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()

    if save_path is not None:
        _save(fig, save_path)
    return fig


# ===================================================================
# Internal helpers
# ===================================================================

def _find_column(df: pd.DataFrame, prefix: str) -> str | None:
    """Find the first column in *df* whose name starts with *prefix*."""
    for col in df.columns:
        if col.startswith(prefix):
            return col
    return None


def _save(fig: plt.Figure, path: str | Path) -> None:
    """Save a figure and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    logger.info("Saved figure to %s", path)
    plt.close(fig)
