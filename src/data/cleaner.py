"""Data cleaning and normalisation pipeline.

Transforms raw OHLCV data into clean, analysis-ready format.
Handles missing data, outliers, and corporate action adjustments.

Usage:
    from src.data.cleaner import DataCleaner
    cleaner = DataCleaner()
    clean_df = cleaner.clean(raw_df, ticker="SHEL.L")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.validators import validate_ohlcv, check_missing_data

logger = get_logger(__name__)


class DataCleaner:
    """Cleans and normalises OHLCV data for downstream analysis.

    Applies a standardised cleaning pipeline:
    1. Sort by date
    2. Remove duplicates
    3. Handle missing data
    4. Detect and handle outliers
    5. Validate output
    """

    def __init__(
        self,
        fill_method: str = "ffill",
        max_missing_pct: float = 0.05,
        outlier_std: float = 5.0,
    ):
        """
        Args:
            fill_method: How to fill missing values ('ffill', 'interpolate', 'drop').
            max_missing_pct: Warn if a column exceeds this missing percentage.
            outlier_std: Flag returns beyond this many standard deviations.
        """
        self.fill_method = fill_method
        self.max_missing_pct = max_missing_pct
        self.outlier_std = outlier_std

    def clean(self, df: pd.DataFrame, ticker: str = "unknown") -> pd.DataFrame:
        """Run the full cleaning pipeline on an OHLCV DataFrame.

        Args:
            df: Raw OHLCV DataFrame with DatetimeIndex.
            ticker: Ticker symbol for logging.

        Returns:
            Cleaned DataFrame.
        """
        logger.info(f"Cleaning {ticker}: {len(df)} rows input")

        df = self._ensure_datetime_index(df, ticker)
        df = self._remove_duplicates(df, ticker)
        df = self._sort_chronologically(df)

        # Check and report missing data before filling
        missing = check_missing_data(df, self.max_missing_pct, ticker)

        df = self._fill_missing(df, ticker)
        df = self._repair_ohlc_consistency(df)
        df = self._handle_outliers(df, ticker)
        df = self._ensure_positive_prices(df)

        # Validate output
        df = validate_ohlcv(df, ticker)

        logger.info(f"Cleaned {ticker}: {len(df)} rows output")
        return df

    def clean_multiple(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Clean multiple tickers.

        Args:
            data: Dictionary of ticker → raw DataFrame.

        Returns:
            Dictionary of ticker → cleaned DataFrame.
        """
        return {ticker: self.clean(df, ticker) for ticker, df in data.items()}

    def _ensure_datetime_index(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Ensure the index is DatetimeIndex."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df = df.set_index("Date")
                df.index = pd.to_datetime(df.index)
            else:
                df.index = pd.to_datetime(df.index)
            logger.debug(f"[{ticker}] Converted index to DatetimeIndex")
        df.index.name = "Date"
        return df

    def _remove_duplicates(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Remove duplicate dates, keeping last occurrence."""
        n_dupes = df.index.duplicated().sum()
        if n_dupes > 0:
            logger.warning(f"[{ticker}] Removing {n_dupes} duplicate dates")
            df = df[~df.index.duplicated(keep="last")]
        return df

    def _sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by date ascending."""
        return df.sort_index()

    def _fill_missing(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Fill missing values according to configured method."""
        n_missing = df.isnull().sum().sum()
        if n_missing == 0:
            return df

        logger.info(f"[{ticker}] Filling {n_missing} missing values ({self.fill_method})")

        if self.fill_method == "ffill":
            df = df.ffill().bfill()  # Forward fill, then back-fill any leading NaNs
        elif self.fill_method == "interpolate":
            df = df.interpolate(method="time").bfill()
        elif self.fill_method == "drop":
            df = df.dropna()
        else:
            raise ValueError(f"Unknown fill method: {self.fill_method}")

        return df

    def _repair_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure High >= max(Open, Close) and Low <= min(Open, Close).

        After forward-filling missing data, OHLC relationships can break.
        This repairs them by adjusting High/Low to be consistent.
        """
        df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
        df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)
        return df

    def _handle_outliers(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Detect and log extreme daily returns (potential data errors).

        Rather than removing outliers automatically (which could hide real
        events like flash crashes), we log warnings for manual review.
        """
        if "Close" not in df.columns or len(df) < 2:
            return df

        returns = df["Close"].pct_change().dropna()
        mean_ret = returns.mean()
        std_ret = returns.std()

        if std_ret == 0:
            return df

        outlier_mask = np.abs(returns - mean_ret) > self.outlier_std * std_ret
        n_outliers = outlier_mask.sum()

        if n_outliers > 0:
            outlier_dates = returns[outlier_mask].index.strftime("%Y-%m-%d").tolist()
            logger.warning(
                f"[{ticker}] {n_outliers} extreme returns detected "
                f"(>{self.outlier_std}σ): {outlier_dates[:5]}..."
            )

        return df

    def _ensure_positive_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip any negative prices to a small positive value."""
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0.01)
        return df


def compute_returns(
    prices: pd.DataFrame | pd.Series,
    windows: list[int] | None = None,
    log_returns: bool = True,
) -> pd.DataFrame:
    """Compute returns over multiple horizons.

    Args:
        prices: Close price DataFrame or Series.
        windows: List of return periods in trading days. Default: [1, 5, 21].
        log_returns: If True, compute log returns. Otherwise simple returns.

    Returns:
        DataFrame with return columns for each window.
    """
    if windows is None:
        windows = [1, 5, 21]

    if isinstance(prices, pd.DataFrame):
        if "Close" in prices.columns:
            prices = prices["Close"]
        else:
            raise ValueError("DataFrame must have a 'Close' column")

    results = {}
    for w in windows:
        if log_returns:
            results[f"ret_{w}d"] = np.log(prices / prices.shift(w))
        else:
            results[f"ret_{w}d"] = prices.pct_change(w)

    return pd.DataFrame(results, index=prices.index)
