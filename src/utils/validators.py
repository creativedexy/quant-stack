"""Data validation utilities — catch problems early.

These validators enforce the project's data conventions:
- DatetimeIndex on all time-series DataFrames
- No future data leakage
- Expected column schemas
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


class DataValidationError(Exception):
    """Raised when data fails validation checks."""


def validate_ohlcv(df: pd.DataFrame, ticker: str = "unknown") -> pd.DataFrame:
    """Validate an OHLCV DataFrame meets project standards.

    Args:
        df: DataFrame to validate.
        ticker: Ticker symbol for error messages.

    Returns:
        The validated DataFrame (unchanged if valid).

    Raises:
        DataValidationError: If validation fails.
    """
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise DataValidationError(
            f"[{ticker}] Missing required columns: {missing}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"[{ticker}] Index must be DatetimeIndex, got {type(df.index).__name__}"
        )

    if not df.index.is_monotonic_increasing:
        raise DataValidationError(
            f"[{ticker}] DatetimeIndex is not sorted in ascending order"
        )

    # Check for negative prices
    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        if (df[col] < 0).any():
            raise DataValidationError(
                f"[{ticker}] Negative values found in {col}"
            )

    # High should be >= Low
    if (df["High"] < df["Low"]).any():
        raise DataValidationError(
            f"[{ticker}] Found rows where High < Low"
        )

    return df


def validate_no_lookahead(
    features: pd.DataFrame,
    target: pd.Series,
    max_date: pd.Timestamp | None = None,
) -> None:
    """Check that features don't contain future information relative to target.

    Args:
        features: Feature DataFrame with DatetimeIndex.
        target: Target Series with DatetimeIndex.
        max_date: If provided, ensure no data beyond this date.

    Raises:
        DataValidationError: If lookahead bias is detected.
    """
    if not isinstance(features.index, pd.DatetimeIndex):
        raise DataValidationError("Features must have DatetimeIndex")

    if not isinstance(target.index, pd.DatetimeIndex):
        raise DataValidationError("Target must have DatetimeIndex")

    if max_date is not None:
        future_features = features.index > max_date
        if future_features.any():
            raise DataValidationError(
                f"Features contain {future_features.sum()} rows after {max_date}"
            )


def check_missing_data(
    df: pd.DataFrame,
    max_missing_pct: float = 0.05,
    ticker: str = "unknown",
) -> dict[str, float]:
    """Report missing data percentages per column.

    Args:
        df: DataFrame to check.
        max_missing_pct: Threshold for raising warnings.
        ticker: Ticker symbol for messages.

    Returns:
        Dictionary of column names to missing data percentages.
    """
    missing_pcts = (df.isnull().sum() / len(df)).to_dict()

    for col, pct in missing_pcts.items():
        if pct > max_missing_pct:
            import warnings
            warnings.warn(
                f"[{ticker}] Column '{col}' has {pct:.1%} missing data "
                f"(threshold: {max_missing_pct:.1%})",
                UserWarning,
                stacklevel=2,
            )

    return missing_pcts
