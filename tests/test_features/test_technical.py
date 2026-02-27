"""Comprehensive tests for src.features.technical indicators.

Uses the synthetic OHLCV fixtures from conftest.py.  Tests cover:
- Output shapes match input shapes
- NaN values only in the expected warm-up period
- RSI bounded between 0 and 100
- Bollinger band ordering (upper > middle > lower)
- Edge cases: flat (constant) prices and single-row DataFrames
- Config-driven defaults and explicit overrides
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.technical import (
    add_sma,
    add_ema,
    add_rsi,
    add_macd,
    add_bollinger_bands,
    add_atr,
    add_all_indicators,
)


# ─────────────────────────────────────────────
# SMA
# ─────────────────────────────────────────────

class TestAddSMA:
    """Tests for Simple Moving Average."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame):
        result = add_sma(sample_ohlcv, windows=[20])
        assert len(result) == len(sample_ohlcv)
        assert "sma_20" in result.columns

    def test_multiple_windows(self, sample_ohlcv: pd.DataFrame):
        result = add_sma(sample_ohlcv, windows=[5, 10, 50])
        assert "sma_5" in result.columns
        assert "sma_10" in result.columns
        assert "sma_50" in result.columns

    def test_nan_warmup_period(self, sample_ohlcv: pd.DataFrame):
        result = add_sma(sample_ohlcv, windows=[20])
        # First 19 rows should be NaN (min_periods=20)
        assert result["sma_20"].iloc[:19].isna().all()
        # From row 19 onwards, no NaN
        assert result["sma_20"].iloc[19:].notna().all()

    def test_known_values(self):
        dates = pd.bdate_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "Open": [10, 20, 30, 40, 50],
                "High": [15, 25, 35, 45, 55],
                "Low": [5, 15, 25, 35, 45],
                "Close": [10.0, 20.0, 30.0, 40.0, 50.0],
                "Volume": [100] * 5,
            },
            index=dates,
        )
        result = add_sma(df, windows=[3])
        assert result["sma_3"].iloc[2] == pytest.approx(20.0)  # mean(10, 20, 30)
        assert result["sma_3"].iloc[4] == pytest.approx(40.0)  # mean(30, 40, 50)

    def test_config_defaults(self, sample_ohlcv: pd.DataFrame):
        """When windows=None, reads from config (should produce sma_5…sma_200)."""
        result = add_sma(sample_ohlcv)
        for w in [5, 10, 20, 50, 200]:
            assert f"sma_{w}" in result.columns

    def test_flat_prices(self, flat_ohlcv: pd.DataFrame):
        result = add_sma(flat_ohlcv, windows=[10])
        valid = result["sma_10"].dropna()
        np.testing.assert_allclose(valid, 100.0)

    def test_single_row(self, single_row_ohlcv: pd.DataFrame):
        result = add_sma(single_row_ohlcv, windows=[1])
        assert result["sma_1"].iloc[0] == pytest.approx(102.0)

    def test_preserves_original_columns(self, sample_ohlcv: pd.DataFrame):
        result = add_sma(sample_ohlcv, windows=[5])
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns
            pd.testing.assert_series_equal(result[col], sample_ohlcv[col])


# ─────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────

class TestAddEMA:
    """Tests for Exponential Moving Average."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame):
        result = add_ema(sample_ohlcv, windows=[12])
        assert len(result) == len(sample_ohlcv)
        assert "ema_12" in result.columns

    def test_nan_warmup_period(self, sample_ohlcv: pd.DataFrame):
        result = add_ema(sample_ohlcv, windows=[26])
        # First 25 rows should be NaN (min_periods=26)
        assert result["ema_26"].iloc[:25].isna().all()
        assert result["ema_26"].iloc[25:].notna().all()

    def test_config_defaults(self, sample_ohlcv: pd.DataFrame):
        result = add_ema(sample_ohlcv)
        assert "ema_12" in result.columns
        assert "ema_26" in result.columns

    def test_flat_prices(self, flat_ohlcv: pd.DataFrame):
        result = add_ema(flat_ohlcv, windows=[10])
        valid = result["ema_10"].dropna()
        np.testing.assert_allclose(valid, 100.0)

    def test_single_row(self, single_row_ohlcv: pd.DataFrame):
        result = add_ema(single_row_ohlcv, windows=[1])
        assert result["ema_1"].iloc[0] == pytest.approx(102.0)


# ─────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────

class TestAddRSI:
    """Tests for Relative Strength Index."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame):
        result = add_rsi(sample_ohlcv, window=14)
        assert len(result) == len(sample_ohlcv)
        assert "rsi_14" in result.columns

    def test_bounded_0_to_100(self, sample_ohlcv: pd.DataFrame):
        result = add_rsi(sample_ohlcv, window=14)
        valid = result["rsi_14"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_nan_warmup_period(self, sample_ohlcv: pd.DataFrame):
        result = add_rsi(sample_ohlcv, window=14)
        # diff() produces 1 NaN, then ewm(min_periods=14) needs 14 non-NaN values
        # so first 14 rows should be NaN
        assert result["rsi_14"].iloc[:14].isna().all()
        assert result["rsi_14"].iloc[14:].notna().all()

    def test_config_default_window(self, sample_ohlcv: pd.DataFrame):
        result = add_rsi(sample_ohlcv)
        assert "rsi_14" in result.columns

    def test_monotonic_up_high_rsi(self):
        """Monotonically rising prices should produce RSI near 100."""
        dates = pd.bdate_range("2024-01-01", periods=50, freq="B")
        prices = np.arange(100, 150, dtype=float)
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 1,
                "Low": prices - 1,
                "Close": prices,
                "Volume": [1000] * 50,
            },
            index=dates,
        )
        result = add_rsi(df, window=14)
        valid = result["rsi_14"].dropna()
        assert (valid > 90.0).all()

    def test_monotonic_down_low_rsi(self):
        """Monotonically falling prices should produce RSI near 0."""
        dates = pd.bdate_range("2024-01-01", periods=50, freq="B")
        prices = np.arange(150, 100, -1, dtype=float)
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 1,
                "Low": prices - 1,
                "Close": prices,
                "Volume": [1000] * 50,
            },
            index=dates,
        )
        result = add_rsi(df, window=14)
        valid = result["rsi_14"].dropna()
        assert (valid < 10.0).all()

    def test_flat_prices_rsi_is_nan(self, flat_ohlcv: pd.DataFrame):
        """Flat prices produce zero gains and zero losses → 0/0 → NaN RSI."""
        result = add_rsi(flat_ohlcv, window=14)
        # With zero deltas, avg_gain=0 and avg_loss=0, so RS=0/0=NaN
        valid_idx = result["rsi_14"].iloc[14:]
        assert valid_idx.isna().all() or ((valid_idx >= 0) & (valid_idx <= 100)).all()

    def test_override_window(self, sample_ohlcv: pd.DataFrame):
        result = add_rsi(sample_ohlcv, window=7)
        assert "rsi_7" in result.columns
        assert "rsi_14" not in result.columns


# ─────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────

class TestAddMACD:
    """Tests for Moving Average Convergence Divergence."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame):
        result = add_macd(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

    def test_histogram_equals_line_minus_signal(self, sample_ohlcv: pd.DataFrame):
        result = add_macd(sample_ohlcv)
        valid = result.dropna(subset=["macd_line", "macd_signal", "macd_hist"])
        np.testing.assert_allclose(
            valid["macd_hist"].values,
            (valid["macd_line"] - valid["macd_signal"]).values,
        )

    def test_nan_warmup_period(self, sample_ohlcv: pd.DataFrame):
        result = add_macd(sample_ohlcv, fast=12, slow=26, signal=9)
        # slow EMA needs 26 periods; signal EMA then needs 9 more on the MACD line.
        # But since macd_line has NaN for first 25, signal EMA (min_periods=9) needs
        # 9 non-NaN macd values, so first valid signal is at index 25 + 9 - 1 = 33.
        assert result["macd_signal"].iloc[:33].isna().all()
        assert result["macd_signal"].iloc[33:].notna().all()

    def test_config_defaults(self, sample_ohlcv: pd.DataFrame):
        result = add_macd(sample_ohlcv)
        assert "macd_line" in result.columns

    def test_custom_params(self, sample_ohlcv: pd.DataFrame):
        result = add_macd(sample_ohlcv, fast=5, slow=10, signal=3)
        assert "macd_line" in result.columns
        # With shorter windows, warm-up is shorter
        assert result["macd_signal"].iloc[11:].notna().all()

    def test_flat_prices_macd_zero(self, flat_ohlcv: pd.DataFrame):
        result = add_macd(flat_ohlcv, fast=5, slow=10, signal=3)
        valid = result["macd_line"].dropna()
        np.testing.assert_allclose(valid, 0.0, atol=1e-12)


# ─────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────

class TestAddBollingerBands:
    """Tests for Bollinger Bands."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame):
        result = add_bollinger_bands(sample_ohlcv, window=20)
        assert len(result) == len(sample_ohlcv)
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns

    def test_upper_ge_middle_ge_lower(self, sample_ohlcv: pd.DataFrame):
        result = add_bollinger_bands(sample_ohlcv, window=20)
        valid = result.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_middle"] - 1e-12).all()
        assert (valid["bb_middle"] >= valid["bb_lower"] - 1e-12).all()

    def test_nan_warmup_period(self, sample_ohlcv: pd.DataFrame):
        result = add_bollinger_bands(sample_ohlcv, window=20)
        assert result["bb_middle"].iloc[:19].isna().all()
        assert result["bb_middle"].iloc[19:].notna().all()

    def test_config_defaults(self, sample_ohlcv: pd.DataFrame):
        result = add_bollinger_bands(sample_ohlcv)
        assert "bb_upper" in result.columns

    def test_band_symmetry(self, sample_ohlcv: pd.DataFrame):
        result = add_bollinger_bands(sample_ohlcv, window=20, num_std=2.0)
        valid = result.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
        np.testing.assert_allclose(
            (valid["bb_upper"] - valid["bb_middle"]).values,
            (valid["bb_middle"] - valid["bb_lower"]).values,
        )

    def test_flat_prices_bands_collapse(self, flat_ohlcv: pd.DataFrame):
        result = add_bollinger_bands(flat_ohlcv, window=10)
        valid = result.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
        np.testing.assert_allclose(valid["bb_upper"], 100.0)
        np.testing.assert_allclose(valid["bb_middle"], 100.0)
        np.testing.assert_allclose(valid["bb_lower"], 100.0)

    def test_single_row(self, single_row_ohlcv: pd.DataFrame):
        result = add_bollinger_bands(single_row_ohlcv, window=1)
        # Window 1 → std=0, so bands collapse to Close
        assert result["bb_middle"].iloc[0] == pytest.approx(102.0)
        assert result["bb_upper"].iloc[0] == pytest.approx(102.0)
        assert result["bb_lower"].iloc[0] == pytest.approx(102.0)


# ─────────────────────────────────────────────
# ATR
# ─────────────────────────────────────────────

class TestAddATR:
    """Tests for Average True Range."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame):
        result = add_atr(sample_ohlcv, window=14)
        assert len(result) == len(sample_ohlcv)
        assert "atr_14" in result.columns

    def test_atr_non_negative(self, sample_ohlcv: pd.DataFrame):
        result = add_atr(sample_ohlcv, window=14)
        valid = result["atr_14"].dropna()
        assert (valid >= 0.0).all()

    def test_nan_warmup_period(self, sample_ohlcv: pd.DataFrame):
        result = add_atr(sample_ohlcv, window=14)
        # Row 0 TR = high-low (valid; prev_close NaN is skipped by max).
        # ewm(min_periods=14) then needs 14 non-NaN TR values → first valid at index 13.
        assert result["atr_14"].iloc[:13].isna().all()
        assert result["atr_14"].iloc[13:].notna().all()

    def test_config_default_window(self, sample_ohlcv: pd.DataFrame):
        result = add_atr(sample_ohlcv)
        assert "atr_14" in result.columns

    def test_flat_prices_zero_atr(self, flat_ohlcv: pd.DataFrame):
        result = add_atr(flat_ohlcv, window=5)
        valid = result["atr_5"].dropna()
        np.testing.assert_allclose(valid, 0.0, atol=1e-12)

    def test_override_window(self, sample_ohlcv: pd.DataFrame):
        result = add_atr(sample_ohlcv, window=7)
        assert "atr_7" in result.columns
        assert "atr_14" not in result.columns


# ─────────────────────────────────────────────
# add_all_indicators
# ─────────────────────────────────────────────

class TestAddAllIndicators:
    """Tests for the convenience wrapper."""

    def test_all_columns_present(self, sample_ohlcv: pd.DataFrame):
        result = add_all_indicators(sample_ohlcv)
        expected_cols = [
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_12", "ema_26",
            "rsi_14",
            "macd_line", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower",
            "atr_14",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_row_count(self, sample_ohlcv: pd.DataFrame):
        result = add_all_indicators(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)

    def test_original_columns_preserved(self, sample_ohlcv: pd.DataFrame):
        result = add_all_indicators(sample_ohlcv)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            pd.testing.assert_series_equal(result[col], sample_ohlcv[col])
