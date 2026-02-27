import numpy as np
import pytest

from quant_stack.indicators import sma, ema, rsi, macd, bollinger_bands


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------

class TestSMA:
    def test_basic(self):
        prices = [1, 2, 3, 4, 5]
        result = sma(prices, window=3)
        expected = [2.0, 3.0, 4.0]
        np.testing.assert_allclose(result, expected)

    def test_window_equals_length(self):
        prices = [10, 20, 30]
        result = sma(prices, window=3)
        np.testing.assert_allclose(result, [20.0])

    def test_window_one(self):
        prices = [5, 10, 15]
        result = sma(prices, window=1)
        np.testing.assert_allclose(result, prices)

    def test_invalid_window_zero(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            sma([1, 2, 3], window=0)

    def test_invalid_window_too_large(self):
        with pytest.raises(ValueError, match="window must be <= length"):
            sma([1, 2], window=5)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestEMA:
    def test_first_value_equals_price(self):
        prices = [100, 110, 105, 115]
        result = ema(prices, span=3)
        assert result[0] == 100.0

    def test_length_matches_input(self):
        prices = [1, 2, 3, 4, 5]
        result = ema(prices, span=3)
        assert len(result) == len(prices)

    def test_span_one_equals_prices(self):
        # alpha = 2/(1+1) = 1.0, so EMA = price itself
        prices = [10, 20, 30, 40]
        result = ema(prices, span=1)
        np.testing.assert_allclose(result, prices)

    def test_known_values(self):
        prices = [1.0, 2.0, 3.0]
        alpha = 2.0 / (3 + 1)  # span=3 -> alpha=0.5
        expected = [1.0, 1.0 * (1 - alpha) + 2.0 * alpha, 0.0]
        expected[2] = expected[1] * (1 - alpha) + 3.0 * alpha
        result = ema(prices, span=3)
        np.testing.assert_allclose(result, expected)

    def test_invalid_span(self):
        with pytest.raises(ValueError, match="span must be >= 1"):
            ema([1, 2], span=0)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_output_length(self):
        prices = list(range(1, 22))  # 21 prices
        result = rsi(prices, window=14)
        assert len(result) == len(prices) - 1 - 14 + 1  # len(deltas) - window + 1

    def test_monotonic_up_is_100(self):
        prices = list(range(100, 120))
        result = rsi(prices, window=14)
        np.testing.assert_allclose(result, 100.0)

    def test_monotonic_down_is_0(self):
        prices = list(range(120, 100, -1))
        result = rsi(prices, window=14)
        np.testing.assert_allclose(result, 0.0)

    def test_range_bounds(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100)) + 100
        result = rsi(prices, window=14)
        assert np.all(result >= 0.0)
        assert np.all(result <= 100.0)

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            rsi([1, 2, 3], window=0)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def test_output_shapes(self):
        prices = np.arange(1, 51, dtype=float)
        macd_line, signal_line, histogram = macd(prices)
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)

    def test_histogram_equals_diff(self):
        prices = np.random.default_rng(0).normal(100, 5, 60)
        macd_line, signal_line, histogram = macd(prices)
        np.testing.assert_allclose(histogram, macd_line - signal_line)

    def test_constant_prices(self):
        prices = [100.0] * 40
        macd_line, signal_line, histogram = macd(prices)
        np.testing.assert_allclose(macd_line, 0.0, atol=1e-12)
        np.testing.assert_allclose(signal_line, 0.0, atol=1e-12)
        np.testing.assert_allclose(histogram, 0.0, atol=1e-12)

    def test_custom_params(self):
        prices = np.arange(1, 31, dtype=float)
        macd_line, signal_line, histogram = macd(
            prices, fast=5, slow=10, signal=3
        )
        # Fast EMA responds quicker, so MACD line should be positive
        # for an uptrending series
        assert macd_line[-1] > 0


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_output_lengths(self):
        prices = list(range(1, 31))
        upper, middle, lower = bollinger_bands(prices, window=20)
        expected_len = len(prices) - 20 + 1
        assert len(upper) == expected_len
        assert len(middle) == expected_len
        assert len(lower) == expected_len

    def test_middle_is_sma(self):
        prices = np.random.default_rng(1).normal(100, 5, 50)
        upper, middle, lower = bollinger_bands(prices, window=20)
        expected_mid = sma(prices, window=20)
        np.testing.assert_allclose(middle, expected_mid)

    def test_band_symmetry(self):
        prices = np.random.default_rng(2).normal(100, 5, 50)
        upper, middle, lower = bollinger_bands(prices, window=20, num_std=2.0)
        np.testing.assert_allclose(upper - middle, middle - lower)

    def test_upper_above_lower(self):
        prices = np.random.default_rng(3).normal(100, 5, 50)
        upper, middle, lower = bollinger_bands(prices, window=20)
        assert np.all(upper >= middle)
        assert np.all(middle >= lower)

    def test_constant_prices_zero_bandwidth(self):
        prices = [50.0] * 30
        upper, middle, lower = bollinger_bands(prices, window=20)
        np.testing.assert_allclose(upper, 50.0)
        np.testing.assert_allclose(middle, 50.0)
        np.testing.assert_allclose(lower, 50.0)
