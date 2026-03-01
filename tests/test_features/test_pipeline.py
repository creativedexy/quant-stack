"""Tests for the composable feature pipeline (src.features.pipeline).

Covers:
- Pipeline produces the correct number of feature columns.
- get_feature_names() matches the actual output columns.
- cutoff_date is respected: no rows after cutoff in output.
- Works for single ticker input.
- Works for dict of multiple tickers (generate returns concat, run returns dict).
- No lookahead bias: features at date T computed only from data at or before T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline


def _has_parquet_engine() -> bool:
    """Check whether a Parquet engine is available."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        pass
    return False


# ─────────────────────────────────────────────
# Expected feature columns from sample_config
# ─────────────────────────────────────────────

# sma_windows: [5, 20, 50]
# ema_windows: [12, 26]
# rsi_window: 14
# macd: macd_line, macd_signal, macd_histogram
# bb: bb_upper, bb_middle, bb_lower, bb_width
# atr_window: 14
# returns.windows: [1, 5, 21]
# volatility_windows: [21, 63]

EXPECTED_FEATURE_COLS = sorted([
    "sma_5", "sma_20", "sma_50",
    "ema_12", "ema_26",
    "rsi_14",
    "macd_line", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "atr_14",
    "ret_1d", "ret_5d", "ret_21d",
    "vol_21d", "vol_63d",
])


# ─────────────────────────────────────────────
# Column count and names
# ─────────────────────────────────────────────

class TestFeatureColumns:
    """Verify the pipeline produces the expected feature set."""

    def test_correct_column_count(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv)
        assert result.shape[1] == len(EXPECTED_FEATURE_COLS)

    def test_all_expected_columns_present(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv)
        for col in EXPECTED_FEATURE_COLS:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_no_ohlcv_columns_in_generate_output(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        """generate() returns features only — no OHLCV columns."""
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col not in result.columns

    def test_get_feature_names_matches_output(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv)
        names = pipe.get_feature_names()
        assert names == sorted(result.columns.tolist())

    def test_get_feature_names_matches_expected(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        pipe.generate(sample_ohlcv)
        assert pipe.get_feature_names() == EXPECTED_FEATURE_COLS

    def test_get_feature_names_sorted(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        pipe.generate(sample_ohlcv)
        names = pipe.get_feature_names()
        assert names == sorted(names)


# ─────────────────────────────────────────────
# cutoff_date
# ─────────────────────────────────────────────

class TestCutoffDate:
    """cutoff_date must be respected."""

    def test_no_rows_after_cutoff(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        cutoff = pd.Timestamp("2017-06-30")
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv, cutoff_date=cutoff)
        assert len(result) > 0
        assert result.index.max() <= cutoff

    def test_cutoff_as_string(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv, cutoff_date="2017-06-30")
        assert result.index.max() <= pd.Timestamp("2017-06-30")

    def test_cutoff_none_uses_all_data(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv, cutoff_date=None)
        # After dropping all-NaN rows the last date should match the input.
        assert result.index.max() == sample_ohlcv.index.max()

    def test_cutoff_before_all_data_returns_empty(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv, cutoff_date="2000-01-01")
        assert len(result) == 0

    def test_cutoff_in_warmup_produces_mostly_nan(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        """A cutoff within the first 50 rows means sma_50 is mostly NaN."""
        start = sample_ohlcv.index[0]
        cutoff = start + pd.Timedelta(days=30)
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv, cutoff_date=cutoff)
        # sma_50 requires 50 data points; with 30 calendar days of data
        # (fewer business days) this column should be entirely NaN.
        if "sma_50" in result.columns and len(result) > 0:
            assert result["sma_50"].isna().all()


# ─────────────────────────────────────────────
# Single ticker
# ─────────────────────────────────────────────

class TestSingleTicker:
    """Pipeline works with a single OHLCV DataFrame."""

    def test_returns_dataframe(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_has_datetime_index(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_no_ticker_column(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        """Single-ticker generate() should not add a 'ticker' column."""
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv)
        assert "ticker" not in result.columns

    def test_output_rows_leq_input(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv)
        assert len(result) <= len(sample_ohlcv)


# ─────────────────────────────────────────────
# Multi-ticker
# ─────────────────────────────────────────────

class TestMultiTicker:
    """Pipeline works with dict[str, DataFrame] input."""

    def test_generate_returns_concat_dataframe(
        self, multi_asset_data: dict[str, pd.DataFrame], sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(multi_asset_data)
        assert isinstance(result, pd.DataFrame)

    def test_generate_has_ticker_column(
        self, multi_asset_data: dict[str, pd.DataFrame], sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(multi_asset_data)
        assert "ticker" in result.columns

    def test_generate_all_tickers_present(
        self, multi_asset_data: dict[str, pd.DataFrame], sample_config: dict,
    ):
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(multi_asset_data)
        assert set(result["ticker"].unique()) == set(multi_asset_data.keys())

    def test_generate_feature_columns_consistent(
        self, multi_asset_data: dict[str, pd.DataFrame], sample_config: dict,
    ):
        """All tickers should have the same feature columns."""
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(multi_asset_data)
        for ticker in multi_asset_data:
            ticker_df = result[result["ticker"] == ticker]
            feature_cols = sorted(
                c for c in ticker_df.columns if c != "ticker"
            )
            assert feature_cols == EXPECTED_FEATURE_COLS

    def test_generate_cutoff_applied_to_all_tickers(
        self, multi_asset_data: dict[str, pd.DataFrame], sample_config: dict,
    ):
        cutoff = pd.Timestamp("2017-06-30")
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(multi_asset_data, cutoff_date=cutoff)
        for ticker in multi_asset_data:
            ticker_df = result[result["ticker"] == ticker]
            if len(ticker_df) > 0:
                assert ticker_df.index.max() <= cutoff, (
                    f"[{ticker}] has data after cutoff"
                )

    def test_run_backward_compat_returns_dict(
        self, multi_asset_data: dict[str, pd.DataFrame], sample_config: dict,
    ):
        """Backward-compatible run() should return dict[str, DataFrame]."""
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.run(multi_asset_data)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(multi_asset_data.keys())

    def test_run_backward_compat_includes_ohlcv(
        self, multi_asset_data: dict[str, pd.DataFrame], sample_config: dict,
    ):
        """Backward-compatible run() should include OHLCV columns."""
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.run(multi_asset_data)
        for ticker, df in result.items():
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                assert col in df.columns, f"[{ticker}] Missing OHLCV column: {col}"


# ─────────────────────────────────────────────
# No lookahead bias
# ─────────────────────────────────────────────

class TestNoLookaheadBias:
    """Features at date T must only use data available at or before T.

    Verification: compute features up to date T, then up to T+30.
    Features at date T must be identical in both runs.
    """

    def test_features_identical_at_shared_dates(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        cutoff_early = pd.Timestamp("2017-06-30")
        cutoff_late = cutoff_early + pd.Timedelta(days=30)

        pipe = FeaturePipeline(config=sample_config)
        early = pipe.generate(sample_ohlcv, cutoff_date=cutoff_early)
        late = pipe.generate(sample_ohlcv, cutoff_date=cutoff_late)

        # Find dates common to both outputs
        common_dates = early.index.intersection(late.index)
        assert len(common_dates) > 0, "No overlapping dates"

        early_common = early.loc[common_dates]
        late_common = late.loc[common_dates]

        # All feature values at shared dates must be identical
        pd.testing.assert_frame_equal(
            early_common, late_common,
            check_exact=False,
            atol=1e-10,
        )

    def test_sma_at_cutoff_matches_manual_calculation(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        """Manually verify SMA-5 at the cutoff date."""
        cutoff = pd.Timestamp("2017-01-31")
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv, cutoff_date=cutoff)

        if cutoff not in result.index:
            # Cutoff may fall on a non-trading day; use the last available date.
            candidates = result.index[result.index <= cutoff]
            if len(candidates) == 0:
                pytest.skip("No data at or before cutoff")
            cutoff = candidates[-1]

        expected_sma5 = sample_ohlcv["Close"].loc[:cutoff].iloc[-5:].mean()
        actual_sma5 = result.loc[cutoff, "sma_5"]
        assert actual_sma5 == pytest.approx(expected_sma5)

    def test_no_future_dates_in_output(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict,
    ):
        """Without an explicit cutoff, dates must not exceed the input range."""
        pipe = FeaturePipeline(config=sample_config)
        result = pipe.generate(sample_ohlcv)
        assert result.index.max() <= sample_ohlcv.index.max()
        assert result.index.min() >= sample_ohlcv.index.min()


# ─────────────────────────────────────────────
# generate_and_save
# ─────────────────────────────────────────────

@pytest.mark.skipif(
    not _has_parquet_engine(),
    reason="pyarrow or fastparquet not installed",
)
class TestGenerateAndSave:
    """Test Parquet persistence."""

    def test_saves_parquet_file(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict, tmp_path,
    ):
        pipe = FeaturePipeline(config=sample_config)
        out = tmp_path / "features.parquet"
        returned = pipe.generate_and_save(sample_ohlcv, out)
        assert returned == out
        assert out.exists()

    def test_saved_matches_generated(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict, tmp_path,
    ):
        pipe = FeaturePipeline(config=sample_config)
        generated = pipe.generate(sample_ohlcv)
        out = tmp_path / "features.parquet"
        pipe.generate_and_save(sample_ohlcv, out)

        loaded = pd.read_parquet(out)
        pd.testing.assert_frame_equal(generated, loaded)

    def test_creates_parent_directories(
        self, sample_ohlcv: pd.DataFrame, sample_config: dict, tmp_path,
    ):
        pipe = FeaturePipeline(config=sample_config)
        out = tmp_path / "sub" / "dir" / "features.parquet"
        pipe.generate_and_save(sample_ohlcv, out)
        assert out.exists()
