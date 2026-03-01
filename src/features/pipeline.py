"""Composable feature pipeline — orchestrates all feature generation.

Takes clean OHLCV data (single ticker or a dict of tickers), computes
technical indicators, return features, and volatility, then produces a
DatetimeIndex-aligned DataFrame ready for modelling.

Usage:
    from src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline()
    features = pipeline.generate(ohlcv_df)
    print(pipeline.get_feature_names())
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.features.technical import compute_all_technical
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Columns that belong to the original OHLCV input, not generated features.
_OHLCV_COLS = {"Open", "High", "Low", "Close", "Volume"}


class FeaturePipeline:
    """Orchestrates feature generation from clean OHLCV data.

    The pipeline delegates to :func:`compute_all_technical` which computes
    SMA, EMA, RSI, MACD, Bollinger Bands, ATR, multi-horizon returns, and
    rolling volatility.

    An explicit ``cutoff_date`` parameter ensures no features are computed
    beyond a given date, preventing lookahead bias in walk-forward pipelines.

    Args:
        config: Full project config dict.  If ``None`` the default
            ``config/settings.yaml`` is loaded.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_config()
        self._config = config
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        cutoff_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Execute the feature pipeline.

        Args:
            data: Either a single OHLCV DataFrame with DatetimeIndex, or a
                ``dict[ticker, DataFrame]`` for multi-ticker processing.
            cutoff_date: If provided, all rows **after** this date are
                dropped *before* returning (but *after* features are
                computed on the truncated input to avoid lookahead bias).

        Returns:
            Feature DataFrame.  For a single ticker this has a
            DatetimeIndex.  For multiple tickers the result is a
            concatenation of all tickers with an additional ``ticker``
            column.
        """
        if isinstance(data, dict):
            frames = []
            for ticker, df in data.items():
                feat = self._generate_single(df, cutoff_date, ticker=ticker)
                if not feat.empty:
                    feat = feat.copy()
                    feat.insert(0, "ticker", ticker)
                    frames.append(feat)
            if not frames:
                self._feature_names = []
                return pd.DataFrame()
            return pd.concat(frames, axis=0)

        return self._generate_single(data, cutoff_date)

    def get_feature_names(self) -> list[str]:
        """Return the names of all generated feature columns.

        Only available after :meth:`generate` has been called at least once.

        Returns:
            Sorted list of feature column names (excludes OHLCV and
            ``ticker`` columns).
        """
        return list(self._feature_names)

    def generate_and_save(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        output_path: str | Path,
        cutoff_date: str | pd.Timestamp | None = None,
    ) -> Path:
        """Generate features and save as Parquet.

        Args:
            data: OHLCV data (single or multi-ticker).
            output_path: Destination file path (``.parquet``).
            cutoff_date: Optional temporal cutoff.

        Returns:
            Path to the saved Parquet file.
        """
        features = self.generate(data, cutoff_date=cutoff_date)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_path)
        logger.info("Saved features to %s (%d rows)", output_path, len(features))
        return output_path

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame | dict[str, pd.DataFrame],
        cutoff_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Backward-compatible wrapper around :meth:`generate`.

        For a single ticker, returns a DataFrame with OHLCV + features.
        For multiple tickers, returns ``dict[ticker, DataFrame]`` (the old
        behaviour).
        """
        if isinstance(data, dict):
            results = {}
            for ticker, df in data.items():
                results[ticker] = self._run_single_compat(df, cutoff_date, ticker)
            return results
        return self._run_single_compat(data, cutoff_date)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_single(
        self,
        df: pd.DataFrame,
        cutoff_date: str | pd.Timestamp | None,
        ticker: str = "unknown",
    ) -> pd.DataFrame:
        """Build features for a single ticker (features-only columns)."""
        # 1. Truncate input BEFORE computing indicators.
        if cutoff_date is not None:
            cutoff = pd.Timestamp(cutoff_date)
            df = df.loc[df.index <= cutoff]
            logger.info(
                "[%s] Truncated to cutoff %s: %d rows",
                ticker, cutoff.date(), len(df),
            )

        if df.empty:
            logger.warning("[%s] Empty DataFrame after cutoff — skipping", ticker)
            self._feature_names = []
            return pd.DataFrame()

        # 2. Compute all features
        features = compute_all_technical(df, config=self._config)

        # 3. Record feature names
        self._feature_names = sorted(features.columns.tolist())

        # 4. Drop rows where ALL feature columns are NaN (warm-up period)
        rows_before = len(features)
        features = features.dropna(how="all")

        logger.info(
            "[%s] Pipeline complete: %d features, %d rows "
            "(dropped %d warm-up rows), date range %s → %s",
            ticker,
            len(self._feature_names),
            len(features),
            rows_before - len(features),
            features.index.min().date() if not features.empty else "N/A",
            features.index.max().date() if not features.empty else "N/A",
        )

        return features

    def _run_single_compat(
        self,
        df: pd.DataFrame,
        cutoff_date: str | pd.Timestamp | None,
        ticker: str = "unknown",
    ) -> pd.DataFrame:
        """Backward-compatible: OHLCV + features, drop rows with NaN in any feature."""
        if cutoff_date is not None:
            cutoff = pd.Timestamp(cutoff_date)
            df = df.loc[df.index <= cutoff]
            logger.info(
                "[%s] Truncated to cutoff %s: %d rows",
                ticker, cutoff.date(), len(df),
            )

        if df.empty:
            logger.warning("[%s] Empty DataFrame after cutoff — skipping", ticker)
            self._feature_names = []
            return df

        features = compute_all_technical(df, config=self._config)

        # Also compute returns via the old cleaner interface for compat
        from src.data.cleaner import compute_returns as _legacy_returns
        ret_cfg = self._config.get("features", {}).get("returns", {})
        legacy_windows = ret_cfg.get("windows", [1, 5, 21])
        legacy_log = ret_cfg.get("log_returns", True)
        returns_df = _legacy_returns(df, windows=legacy_windows, log_returns=legacy_log)

        # Join everything: OHLCV + technical features + legacy returns
        # (avoid duplicate return columns)
        existing_ret_cols = [c for c in returns_df.columns if c in features.columns]
        returns_df = returns_df.drop(columns=existing_ret_cols, errors="ignore")

        out = df.join(features)
        if not returns_df.empty and not returns_df.columns.empty:
            out = out.join(returns_df)

        self._feature_names = sorted(
            col for col in out.columns if col not in _OHLCV_COLS
        )

        rows_before = len(out)
        out = out.dropna(subset=self._feature_names)

        logger.info(
            "[%s] Pipeline (compat) complete: %d features, %d rows "
            "(dropped %d warm-up rows), date range %s → %s",
            ticker,
            len(self._feature_names),
            len(out),
            rows_before - len(out),
            out.index.min().date() if not out.empty else "N/A",
            out.index.max().date() if not out.empty else "N/A",
        )

        return out
