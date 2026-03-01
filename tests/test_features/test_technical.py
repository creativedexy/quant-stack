"""Comprehensive tests for src.features.technical indicators.

Uses the synthetic OHLCV fixtures from conftest.py.  Tests cover:
- Output shape matches input length
- Column names are correct
- No NaN values beyond expected warm-up period
- RSI is bounded between 0 and 100
- Bollinger band ordering (upper > middle > lower)
- ATR is always positive
- MACD histogram = MACD line - signal line
- Reproducibility: same input → same output
- Edge cases: flat prices, single-row input
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.technical import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_atr,
    compute_returns,
    compute_volatility,
    compute_all_technical,
    # Backward-compat wrappers
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

class TestComputeSMA:
    """Tests for Simple Moving Average."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_sma(sample_ohlcv, windows=[20])
        assert len(result) == len(sample_ohlcv)
        assert result.columns.tolist() == ["sma_20"]

    def test_multiple_windows(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_sma(sample_ohlcv, windows=[5, 10, 50])
        assert "sma_5" in result.columns
        assert "sma_10" in result.columns
        assert "sma_50" in result.columns

    def test_nan_warmup_period(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_sma(sample_ohlcv, windows=[20])
        assert result["sma_20"].iloc[:19].isna().all()
        assert result["sma_20"].iloc[19:].notna().all()

    def test_known_values(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"Close": [10.0, 20.0, 30.0, 40.0, 50.0],
             "Open": [10, 20, 30, 40, 50],
             "High": [15, 25, 35, 45, 55],
             "Low": [5, 15, 25, 35, 45],
             "Volume": [100] * 5},
            index=dates,
        )
        result = compute_sma(df, windows=[3])
        assert result["sma_3"].iloc[2] == pytest.approx(20.0)
        assert result["sma_3"].iloc[4] == pytest.approx(40.0)

    def test_config_defaults(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_sma(sample_ohlcv)
        for w in [5, 10, 20, 50, 200]:
            assert f"sma_{w}" in result.columns

    def test_flat_prices(self, flat_ohlcv: pd.DataFrame) -> None:
        result = compute_sma(flat_ohlcv, windows=[10])
        valid = result["sma_10"].dropna()
        np.testing.assert_allclose(valid, 100.0)

    def test_single_row(self, single_row_ohlcv: pd.DataFrame) -> None:
        result = compute_sma(single_row_ohlcv, windows=[1])
        assert isinstance(result, pd.DataFrame)
        assert result["sma_1"].iloc[0] == pytest.approx(102.0)

    def test_does_not_include_ohlcv(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_sma(sample_ohlcv, windows=[5])
        assert "Close" not in result.columns

    def test_reproducibility(self, sample_ohlcv: pd.DataFrame) -> None:
        r1 = compute_sma(sample_ohlcv, windows=[20])
        r2 = compute_sma(sample_ohlcv, windows=[20])
        pd.testing.assert_frame_equal(r1, r2)


# ─────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────

class TestComputeEMA:
    """Tests for Exponential Moving Average."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_ema(sample_ohlcv, windows=[12])
        assert len(result) == len(sample_ohlcv)
        assert "ema_12" in result.columns

    def test_nan_warmup_period(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_ema(sample_ohlcv, windows=[26])
        assert result["ema_26"].iloc[:25].isna().all()
        assert result["ema_26"].iloc[25:].notna().all()

    def test_config_defaults(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_ema(sample_ohlcv)
        assert "ema_12" in result.columns
        assert "ema_26" in result.columns

    def test_flat_prices(self, flat_ohlcv: pd.DataFrame) -> None:
        result = compute_ema(flat_ohlcv, windows=[10])
        valid = result["ema_10"].dropna()
        np.testing.assert_allclose(valid, 100.0)

    def test_single_row(self, single_row_ohlcv: pd.DataFrame) -> None:
        result = compute_ema(single_row_ohlcv, windows=[1])
        assert isinstance(result, pd.DataFrame)
        assert result["ema_1"].iloc[0] == pytest.approx(102.0)

    def test_reproducibility(self, sample_ohlcv: pd.DataFrame) -> None:
        r1 = compute_ema(sample_ohlcv, windows=[12])
        r2 = compute_ema(sample_ohlcv, windows=[12])
        pd.testing.assert_frame_equal(r1, r2)


# ─────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────

class TestComputeRSI:
    """Tests for Relative Strength Index."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_rsi(sample_ohlcv, window=14)
        assert len(result) == len(sample_ohlcv)
        assert "rsi_14" in result.columns

    def test_bounded_0_to_100(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_rsi(sample_ohlcv, window=14)
        valid = result["rsi_14"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_nan_warmup_period(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_rsi(sample_ohlcv, window=14)
        assert result["rsi_14"].iloc[:14].isna().all()
        assert result["rsi_14"].iloc[14:].notna().all()

    def test_config_default_window(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_rsi(sample_ohlcv)
        assert "rsi_14" in result.columns

    def test_monotonic_up_high_rsi(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=50, freq="B")
        prices = np.arange(100, 150, dtype=float)
        df = pd.DataFrame(
            {"Open": prices, "High": prices + 1, "Low": prices - 1,
             "Close": prices, "Volume": [1000] * 50},
            index=dates,
        )
        result = compute_rsi(df, window=14)
        valid = result["rsi_14"].dropna()
        assert (valid > 90.0).all()

    def test_monotonic_down_low_rsi(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=50, freq="B")
        prices = np.arange(150, 100, -1, dtype=float)
        df = pd.DataFrame(
            {"Open": prices, "High": prices + 1, "Low": prices - 1,
             "Close": prices, "Volume": [1000] * 50},
            index=dates,
        )
        result = compute_rsi(df, window=14)
        valid = result["rsi_14"].dropna()
        assert (valid < 10.0).all()

    def test_flat_prices_does_not_crash(self, flat_ohlcv: pd.DataFrame) -> None:
        """Flat prices → zero gain/loss → NaN RSI.  Must not crash."""
        result = compute_rsi(flat_ohlcv, window=14)
        assert isinstance(result, pd.DataFrame)
        valid_idx = result["rsi_14"].iloc[14:]
        # May be NaN (0/0) or 50 depending on implementation — both acceptable
        assert valid_idx.isna().all() or ((valid_idx >= 0) & (valid_idx <= 100)).all()

    def test_override_window(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_rsi(sample_ohlcv, window=7)
        assert "rsi_7" in result.columns
        assert "rsi_14" not in result.columns

    def test_single_row(self, single_row_ohlcv: pd.DataFrame) -> None:
        result = compute_rsi(single_row_ohlcv, window=14)
        assert isinstance(result, pd.DataFrame)
        assert result["rsi_14"].isna().all()

    def test_reproducibility(self, sample_ohlcv: pd.DataFrame) -> None:
        r1 = compute_rsi(sample_ohlcv, window=14)
        r2 = compute_rsi(sample_ohlcv, window=14)
        pd.testing.assert_frame_equal(r1, r2)


# ─────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────

class TestComputeMACD:
    """Tests for Moving Average Convergence Divergence."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_macd(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_histogram_equals_line_minus_signal(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_macd(sample_ohlcv)
        valid = result.dropna()
        np.testing.assert_allclose(
            valid["macd_histogram"].values,
            (valid["macd_line"] - valid["macd_signal"]).values,
        )

    def test_nan_warmup_period(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_macd(sample_ohlcv, fast=12, slow=26, signal=9)
        assert result["macd_signal"].iloc[:33].isna().all()
        assert result["macd_signal"].iloc[33:].notna().all()

    def test_config_defaults(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_macd(sample_ohlcv)
        assert "macd_line" in result.columns

    def test_custom_params(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_macd(sample_ohlcv, fast=5, slow=10, signal=3)
        assert result["macd_signal"].iloc[11:].notna().all()

    def test_flat_prices_macd_zero(self, flat_ohlcv: pd.DataFrame) -> None:
        result = compute_macd(flat_ohlcv, fast=5, slow=10, signal=3)
        valid = result["macd_line"].dropna()
        np.testing.assert_allclose(valid, 0.0, atol=1e-12)

    def test_single_row(self, single_row_ohlcv: pd.DataFrame) -> None:
        result = compute_macd(single_row_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_reproducibility(self, sample_ohlcv: pd.DataFrame) -> None:
        r1 = compute_macd(sample_ohlcv)
        r2 = compute_macd(sample_ohlcv)
        pd.testing.assert_frame_equal(r1, r2)


# ─────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────

class TestComputeBollingerBands:
    """Tests for Bollinger Bands."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_bollinger_bands(sample_ohlcv, window=20)
        assert len(result) == len(sample_ohlcv)
        for col in ["bb_upper", "bb_middle", "bb_lower", "bb_width"]:
            assert col in result.columns

    def test_upper_ge_middle_ge_lower(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_bollinger_bands(sample_ohlcv, window=20)
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"] - 1e-12).all()
        assert (valid["bb_middle"] >= valid["bb_lower"] - 1e-12).all()

    def test_nan_warmup_period(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_bollinger_bands(sample_ohlcv, window=20)
        assert result["bb_middle"].iloc[:19].isna().all()
        assert result["bb_middle"].iloc[19:].notna().all()

    def test_config_defaults(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_bollinger_bands(sample_ohlcv)
        assert "bb_upper" in result.columns

    def test_band_symmetry(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_bollinger_bands(sample_ohlcv, window=20, num_std=2.0)
        valid = result.dropna()
        np.testing.assert_allclose(
            (valid["bb_upper"] - valid["bb_middle"]).values,
            (valid["bb_middle"] - valid["bb_lower"]).values,
        )

    def test_flat_prices_bands_collapse(self, flat_ohlcv: pd.DataFrame) -> None:
        result = compute_bollinger_bands(flat_ohlcv, window=10)
        valid = result.dropna()
        np.testing.assert_allclose(valid["bb_upper"], 100.0)
        np.testing.assert_allclose(valid["bb_middle"], 100.0)
        np.testing.assert_allclose(valid["bb_lower"], 100.0)

    def test_bb_width_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_bollinger_bands(sample_ohlcv, window=20)
        valid = result["bb_width"].dropna()
        assert (valid >= 0.0).all()

    def test_single_row(self, single_row_ohlcv: pd.DataFrame) -> None:
        result = compute_bollinger_bands(single_row_ohlcv, window=1)
        assert isinstance(result, pd.DataFrame)

    def test_reproducibility(self, sample_ohlcv: pd.DataFrame) -> None:
        r1 = compute_bollinger_bands(sample_ohlcv, window=20)
        r2 = compute_bollinger_bands(sample_ohlcv, window=20)
        pd.testing.assert_frame_equal(r1, r2)


# ─────────────────────────────────────────────
# ATR
# ─────────────────────────────────────────────

class TestComputeATR:
    """Tests for Average True Range."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_atr(sample_ohlcv, window=14)
        assert len(result) == len(sample_ohlcv)
        assert "atr_14" in result.columns

    def test_atr_always_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_atr(sample_ohlcv, window=14)
        valid = result["atr_14"].dropna()
        assert (valid >= 0.0).all()

    def test_nan_warmup_period(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_atr(sample_ohlcv, window=14)
        assert result["atr_14"].iloc[:13].isna().all()
        assert result["atr_14"].iloc[13:].notna().all()

    def test_config_default_window(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_atr(sample_ohlcv)
        assert "atr_14" in result.columns

    def test_flat_prices_zero_atr(self, flat_ohlcv: pd.DataFrame) -> None:
        result = compute_atr(flat_ohlcv, window=5)
        valid = result["atr_5"].dropna()
        np.testing.assert_allclose(valid, 0.0, atol=1e-12)

    def test_override_window(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_atr(sample_ohlcv, window=7)
        assert "atr_7" in result.columns
        assert "atr_14" not in result.columns

    def test_single_row(self, single_row_ohlcv: pd.DataFrame) -> None:
        result = compute_atr(single_row_ohlcv, window=14)
        assert isinstance(result, pd.DataFrame)

    def test_reproducibility(self, sample_ohlcv: pd.DataFrame) -> None:
        r1 = compute_atr(sample_ohlcv, window=14)
        r2 = compute_atr(sample_ohlcv, window=14)
        pd.testing.assert_frame_equal(r1, r2)


# ─────────────────────────────────────────────
# Returns
# ─────────────────────────────────────────────

class TestComputeReturns:
    """Tests for multi-horizon return computation."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_returns(sample_ohlcv, windows=[1, 5, 21])
        assert len(result) == len(sample_ohlcv)
        assert "ret_1d" in result.columns
        assert "ret_5d" in result.columns
        assert "ret_21d" in result.columns

    def test_config_defaults(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_returns(sample_ohlcv)
        for w in [1, 5, 10, 21, 63, 252]:
            assert f"ret_{w}d" in result.columns

    def test_nan_first_w_rows(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_returns(sample_ohlcv, windows=[5])
        assert result["ret_5d"].iloc[:5].isna().all()
        assert result["ret_5d"].iloc[5:].notna().all()

    def test_log_returns(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_returns(sample_ohlcv, windows=[1], log_returns=True)
        close = sample_ohlcv["Close"]
        expected = np.log(close / close.shift(1))
        pd.testing.assert_series_equal(result["ret_1d"], expected, check_names=False)

    def test_simple_returns(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_returns(sample_ohlcv, windows=[1], log_returns=False)
        close = sample_ohlcv["Close"]
        expected = close.pct_change(1)
        pd.testing.assert_series_equal(result["ret_1d"], expected, check_names=False)

    def test_flat_prices_zero_returns(self, flat_ohlcv: pd.DataFrame) -> None:
        result = compute_returns(flat_ohlcv, windows=[1], log_returns=True)
        valid = result["ret_1d"].dropna()
        np.testing.assert_allclose(valid, 0.0, atol=1e-12)

    def test_single_row(self, single_row_ohlcv: pd.DataFrame) -> None:
        result = compute_returns(single_row_ohlcv, windows=[1])
        assert isinstance(result, pd.DataFrame)

    def test_reproducibility(self, sample_ohlcv: pd.DataFrame) -> None:
        r1 = compute_returns(sample_ohlcv, windows=[1, 5])
        r2 = compute_returns(sample_ohlcv, windows=[1, 5])
        pd.testing.assert_frame_equal(r1, r2)


# ─────────────────────────────────────────────
# Volatility
# ─────────────────────────────────────────────

class TestComputeVolatility:
    """Tests for annualised rolling volatility."""

    def test_output_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_volatility(sample_ohlcv, windows=[21])
        assert len(result) == len(sample_ohlcv)
        assert "vol_21d" in result.columns

    def test_config_defaults(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_volatility(sample_ohlcv)
        assert "vol_21d" in result.columns
        assert "vol_63d" in result.columns

    def test_nan_warmup(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_volatility(sample_ohlcv, windows=[21])
        # Need 1 row for log return diff + 21 rows for rolling = 22 NaN rows
        # (shift(1) makes first row NaN, then rolling needs 21 non-NaN)
        assert result["vol_21d"].iloc[:21].isna().all()
        assert result["vol_21d"].iloc[21:].notna().all()

    def test_always_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_volatility(sample_ohlcv, windows=[21])
        valid = result["vol_21d"].dropna()
        assert (valid >= 0.0).all()

    def test_annualised(self, sample_ohlcv: pd.DataFrame) -> None:
        """Volatility should be annualised (multiplied by sqrt(252))."""
        result = compute_volatility(sample_ohlcv, windows=[21])
        # A typical stock has annualised vol between 10% and 80%
        valid = result["vol_21d"].dropna()
        assert valid.median() > 0.05  # > 5% annualised
        assert valid.median() < 1.0   # < 100% annualised

    def test_flat_prices_zero_vol(self, flat_ohlcv: pd.DataFrame) -> None:
        result = compute_volatility(flat_ohlcv, windows=[5])
        valid = result["vol_5d"].dropna()
        np.testing.assert_allclose(valid, 0.0, atol=1e-12)

    def test_single_row(self, single_row_ohlcv: pd.DataFrame) -> None:
        result = compute_volatility(single_row_ohlcv, windows=[21])
        assert isinstance(result, pd.DataFrame)


# ─────────────────────────────────────────────
# compute_all_technical
# ─────────────────────────────────────────────

class TestComputeAllTechnical:
    """Tests for the convenience aggregator."""

    def test_no_ohlcv_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_all_technical(sample_ohlcv)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col not in result.columns

    def test_has_all_indicator_families(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_all_technical(sample_ohlcv)
        # Check a sample from each family
        assert any(c.startswith("sma_") for c in result.columns)
        assert any(c.startswith("ema_") for c in result.columns)
        assert any(c.startswith("rsi_") for c in result.columns)
        assert "macd_line" in result.columns
        assert "macd_histogram" in result.columns
        assert "bb_upper" in result.columns
        assert "bb_width" in result.columns
        assert any(c.startswith("atr_") for c in result.columns)
        assert any(c.startswith("ret_") for c in result.columns)
        assert any(c.startswith("vol_") for c in result.columns)

    def test_output_row_count(self, sample_ohlcv: pd.DataFrame) -> None:
        result = compute_all_technical(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)

    def test_reproducibility(self, sample_ohlcv: pd.DataFrame) -> None:
        r1 = compute_all_technical(sample_ohlcv)
        r2 = compute_all_technical(sample_ohlcv)
        pd.testing.assert_frame_equal(r1, r2)


# ─────────────────────────────────────────────
# Backward-compat add_* wrappers
# ─────────────────────────────────────────────

class TestAddWrappers:
    """Verify backward-compatible add_* functions still work."""

    def test_add_sma_preserves_ohlcv(self, sample_ohlcv: pd.DataFrame) -> None:
        result = add_sma(sample_ohlcv, windows=[20])
        assert "Close" in result.columns
        assert "sma_20" in result.columns
        pd.testing.assert_series_equal(result["Close"], sample_ohlcv["Close"])

    def test_add_all_indicators(self, sample_ohlcv: pd.DataFrame) -> None:
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
        # OHLCV preserved
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            pd.testing.assert_series_equal(result[col], sample_ohlcv[col])
