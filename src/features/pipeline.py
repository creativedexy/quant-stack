"""Composable feature pipeline — orchestrates all feature generation.

Takes clean OHLCV data (single ticker or a dict of tickers), applies
technical indicators and return features, then produces a single
DatetimeIndex-aligned DataFrame ready for modelling.

Usage:
    from src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline()
    features = pipeline.run(ohlcv_df)
    print(pipeline.get_feature_names())
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.data.cleaner import compute_returns
from src.features.technical import add_all_indicators
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Columns that belong to the original OHLCV input, not generated features.
_OHLCV_COLS = {"Open", "High", "Low", "Close", "Volume"}


class FeaturePipeline:
    """Orchestrates feature generation from clean OHLCV data.

    The pipeline applies two stages in order:
    1. **Technical indicators** — SMA, EMA, RSI, MACD, Bollinger Bands, ATR
       (via ``src.features.technical.add_all_indicators``).
    2. **Return features** — log or simple returns over multiple horizons
       (via ``src.data.cleaner.compute_returns``).

    An explicit ``cutoff_date`` parameter ensures no features are computed
    beyond a given date, preventing lookahead bias in walk-forward pipelines.

    Args:
        return_windows: Horizons (in trading days) for return features.
            Defaults to config ``features.returns.windows``.
        log_returns: Whether to compute log returns.
            Defaults to config ``features.returns.log_returns``.
        drop_na: If ``True`` (default), rows containing NaN from indicator
            warm-up periods are dropped.  Set to ``False`` to keep them.
    """

    def __init__(
        self,
        return_windows: Sequence[int] | None = None,
        log_returns: bool | None = None,
        drop_na: bool = True,
    ) -> None:
        cfg = load_config().get("features", {}).get("returns", {})
        self.return_windows: list[int] = (
            list(return_windows) if return_windows is not None
            else cfg.get("windows", [1, 5, 21])
        )
        self.log_returns: bool = (
            log_returns if log_returns is not None
            else cfg.get("log_returns", True)
        )
        self.drop_na = drop_na

        # Populated after run()
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        cutoff_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Execute the feature pipeline.

        Args:
            data: Either a single OHLCV DataFrame with DatetimeIndex, or a
                ``dict[ticker, DataFrame]`` for multi-ticker processing.
            cutoff_date: If provided, all input data is truncated to this date
                *before* any features are computed.  This guarantees that
                indicators cannot peek beyond the cutoff.

        Returns:
            A feature DataFrame (single ticker) or a ``dict[ticker, DataFrame]``
            (multi-ticker), containing the original OHLCV columns plus all
            generated features.
        """
        if isinstance(data, dict):
            results = {}
            for ticker, df in data.items():
                results[ticker] = self._run_single(df, cutoff_date, ticker=ticker)
            return results
        return self._run_single(data, cutoff_date)

    def get_feature_names(self) -> list[str]:
        """Return the names of all generated feature columns.

        Only available after :meth:`run` has been called at least once.

        Returns:
            Sorted list of feature column names (excludes OHLCV columns).
        """
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_single(
        self,
        df: pd.DataFrame,
        cutoff_date: str | pd.Timestamp | None,
        ticker: str = "unknown",
    ) -> pd.DataFrame:
        """Build features for a single ticker."""
        # 1. Truncate to cutoff_date BEFORE computing anything.
        if cutoff_date is not None:
            cutoff = pd.Timestamp(cutoff_date)
            df = df.loc[df.index <= cutoff]
            logger.info(
                f"[{ticker}] Truncated to cutoff {cutoff.date()}: {len(df)} rows"
            )

        if df.empty:
            logger.warning(f"[{ticker}] Empty DataFrame after cutoff — skipping")
            self._feature_names = []
            return df

        # 2. Technical indicators
        out = add_all_indicators(df)

        # 3. Return features
        returns_df = compute_returns(
            df, windows=self.return_windows, log_returns=self.log_returns,
        )
        out = out.join(returns_df)

        # 4. Record feature column names (everything that is not OHLCV)
        self._feature_names = sorted(
            col for col in out.columns if col not in _OHLCV_COLS
        )

        # 5. Optionally drop warm-up NaN rows
        rows_before = len(out)
        if self.drop_na:
            out = out.dropna(subset=self._feature_names)

        logger.info(
            f"[{ticker}] Pipeline complete: {len(self._feature_names)} features, "
            f"{len(out)} rows "
            f"({'dropped ' + str(rows_before - len(out)) + ' warm-up rows' if self.drop_na else 'NaN rows kept'}), "
            f"date range {out.index.min().date()} → {out.index.max().date()}"
        )

        return out
