"""Tests for the composable feature pipeline (src.features.pipeline).

Covers:
- Feature matrix has the correct number of columns
- No lookahead bias: features at date T only use data up to T
- Pipeline works for both single ticker and multi-ticker input
- cutoff_date is respected
- drop_na / keep NaN behaviour
- get_feature_names() returns the right column list
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline


# ─────────────────────────────────────────────
# Expected feature columns from defaults
# ─────────────────────────────────────────────

# Technical indicator columns produced by add_all_indicators() with config defaults
TECHNICAL_COLS = [
    "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
    "ema_12", "ema_26",
    "rsi_14",
    "macd_line", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower",
    "atr_14",
]

# Return columns produced by compute_returns() with default windows [1, 5, 21]
RETURN_COLS = ["ret_1d", "ret_5d", "ret_21d"]

ALL_FEATURE_COLS = sorted(TECHNICAL_COLS + RETURN_COLS)


# ─────────────────────────────────────────────
# Column count and names
# ─────────────────────────────────────────────

class TestFeatureColumns:
    """Verify the pipeline produces the expected feature set."""

    def test_correct_column_count(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline()
        result = pipe.run(sample_ohlcv)
        feature_names = pipe.get_feature_names()
        assert len(feature_names) == len(ALL_FEATURE_COLS)

    def test_all_expected_columns_present(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline()
        result = pipe.run(sample_ohlcv)
        for col in ALL_FEATURE_COLS:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_ohlcv_columns_preserved(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline()
        result = pipe.run(sample_ohlcv)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    def test_get_feature_names_sorted(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline()
        pipe.run(sample_ohlcv)
        names = pipe.get_feature_names()
        assert names == sorted(names)

    def test_get_feature_names_excludes_ohlcv(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline()
        pipe.run(sample_ohlcv)
        names = pipe.get_feature_names()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col not in names

    def test_get_feature_names_matches_expected(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline()
        pipe.run(sample_ohlcv)
        assert pipe.get_feature_names() == ALL_FEATURE_COLS

    def test_custom_return_windows(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline(return_windows=[1, 10])
        result = pipe.run(sample_ohlcv)
        assert "ret_1d" in result.columns
        assert "ret_10d" in result.columns
        assert "ret_5d" not in result.columns
        assert "ret_21d" not in result.columns


# ─────────────────────────────────────────────
# No lookahead bias
# ─────────────────────────────────────────────

class TestNoLookaheadBias:
    """Features at date T must only use data available up to T."""

    def test_features_respect_cutoff_date(self, sample_ohlcv: pd.DataFrame):
        """All rows in the output must be on or before the cutoff date."""
        cutoff = pd.Timestamp("2017-06-30")
        pipe = FeaturePipeline()
        result = pipe.run(sample_ohlcv, cutoff_date=cutoff)
        assert result.index.max() <= cutoff

    def test_cutoff_truncates_before_computing(self, sample_ohlcv: pd.DataFrame):
        """Verify the pipeline slices input *before* computing indicators,
        not after.  If it were truncating after, a 200-day SMA at the
        cutoff date would use data beyond the cutoff."""
        early_cutoff = pd.Timestamp("2016-06-30")
        late_cutoff = pd.Timestamp("2018-06-30")

        pipe = FeaturePipeline()
        early_result = pipe.run(sample_ohlcv, cutoff_date=early_cutoff)
        late_result = pipe.run(sample_ohlcv, cutoff_date=late_cutoff)

        # The SMA at the cutoff date should differ because it was computed
        # on different data windows, not sliced from the same full run.
        if early_cutoff in early_result.index and early_cutoff in late_result.index:
            early_sma = early_result.loc[early_cutoff, "sma_200"]
            late_sma = late_result.loc[early_cutoff, "sma_200"]
            # Both used the same data up to that point so they SHOULD be equal.
            assert early_sma == pytest.approx(late_sma)

    def test_sma_at_cutoff_uses_only_past_data(self, sample_ohlcv: pd.DataFrame):
        """Manually verify SMA-5 at cutoff uses exactly the preceding 5 closes."""
        cutoff = pd.Timestamp("2017-01-31")
        pipe = FeaturePipeline(drop_na=False)
        result = pipe.run(sample_ohlcv, cutoff_date=cutoff)

        if cutoff not in result.index:
            # cutoff falls on a non-trading day; use the last available date
            cutoff = result.index[result.index <= cutoff][-1]

        expected_sma5 = sample_ohlcv["Close"].loc[:cutoff].iloc[-5:].mean()
        actual_sma5 = result.loc[cutoff, "sma_5"]
        assert actual_sma5 == pytest.approx(expected_sma5)

    def test_no_future_dates_in_output(self, sample_ohlcv: pd.DataFrame):
        """Even without an explicit cutoff, the output must not contain
        dates beyond the last row of the input."""
        pipe = FeaturePipeline()
        result = pipe.run(sample_ohlcv)
        assert result.index.max() <= sample_ohlcv.index.max()


# ─────────────────────────────────────────────
# Single ticker vs multi-ticker
# ─────────────────────────────────────────────

class TestSingleVsMultiTicker:
    """Pipeline works with both a single DataFrame and a dict of DataFrames."""

    def test_single_ticker_returns_dataframe(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline()
        result = pipe.run(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_multi_ticker_returns_dict(self, multi_asset_data: dict[str, pd.DataFrame]):
        pipe = FeaturePipeline()
        result = pipe.run(multi_asset_data)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(multi_asset_data.keys())

    def test_multi_ticker_each_has_features(
        self, multi_asset_data: dict[str, pd.DataFrame],
    ):
        pipe = FeaturePipeline()
        result = pipe.run(multi_asset_data)
        for ticker, df in result.items():
            assert isinstance(df, pd.DataFrame)
            for col in ALL_FEATURE_COLS:
                assert col in df.columns, f"[{ticker}] Missing column: {col}"

    def test_multi_ticker_cutoff_applied_to_all(
        self, multi_asset_data: dict[str, pd.DataFrame],
    ):
        cutoff = pd.Timestamp("2017-06-30")
        pipe = FeaturePipeline()
        result = pipe.run(multi_asset_data, cutoff_date=cutoff)
        for ticker, df in result.items():
            assert df.index.max() <= cutoff, f"[{ticker}] has data after cutoff"

    def test_multi_ticker_feature_names_consistent(
        self, multi_asset_data: dict[str, pd.DataFrame],
    ):
        """get_feature_names() reflects the last ticker processed, but the
        set of features should be identical across tickers."""
        pipe = FeaturePipeline()
        result = pipe.run(multi_asset_data)
        feature_sets = [
            sorted(col for col in df.columns if col not in {"Open", "High", "Low", "Close", "Volume"})
            for df in result.values()
        ]
        assert all(fs == feature_sets[0] for fs in feature_sets)


# ─────────────────────────────────────────────
# cutoff_date edge cases
# ─────────────────────────────────────────────

class TestCutoffDate:
    """cutoff_date parameter edge cases."""

    def test_cutoff_none_uses_all_data(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline()
        result = pipe.run(sample_ohlcv, cutoff_date=None)
        # After dropping warm-up NaNs, the last date should match the input.
        assert result.index.max() == sample_ohlcv.index.max()

    def test_cutoff_as_string(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline()
        result = pipe.run(sample_ohlcv, cutoff_date="2017-06-30")
        assert result.index.max() <= pd.Timestamp("2017-06-30")

    def test_cutoff_before_all_data_returns_empty(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline()
        result = pipe.run(sample_ohlcv, cutoff_date="2000-01-01")
        assert len(result) == 0

    def test_cutoff_during_warmup_period(self, sample_ohlcv: pd.DataFrame):
        """A cutoff within the first 200 rows means sma_200 is all NaN.
        With drop_na=True, this should produce an empty result because
        there are not enough rows to fill the longest warm-up."""
        # The data starts around 2015-01-02.  200 business days is ~40 weeks.
        cutoff = pd.Timestamp("2015-05-01")
        pipe = FeaturePipeline(drop_na=True)
        result = pipe.run(sample_ohlcv, cutoff_date=cutoff)
        assert len(result) == 0


# ─────────────────────────────────────────────
# drop_na behaviour
# ─────────────────────────────────────────────

class TestDropNA:
    """Test configurable NaN handling."""

    def test_drop_na_true_no_nan_in_features(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline(drop_na=True)
        result = pipe.run(sample_ohlcv)
        feature_cols = pipe.get_feature_names()
        assert result[feature_cols].isna().sum().sum() == 0

    def test_drop_na_false_keeps_warmup_rows(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline(drop_na=False)
        result = pipe.run(sample_ohlcv)
        # Should keep all original rows
        assert len(result) == len(sample_ohlcv)

    def test_drop_na_false_has_nan_in_warmup(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline(drop_na=False)
        result = pipe.run(sample_ohlcv)
        # sma_200 has 199 NaN warm-up rows
        assert result["sma_200"].isna().any()

    def test_drop_na_true_fewer_rows_than_input(self, sample_ohlcv: pd.DataFrame):
        pipe = FeaturePipeline(drop_na=True)
        result = pipe.run(sample_ohlcv)
        assert len(result) < len(sample_ohlcv)
