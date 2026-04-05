"""Tests for src.services.chart_service.ChartService.

Nine tests covering all five public methods plus edge cases.
"""

from __future__ import annotations

import pytest

from src.services.chart_service import ChartService


@pytest.fixture()
def chart_svc() -> ChartService:
    """Create a ChartService with default config."""
    config = {
        "universe": {"benchmark": "^FTSE"},
        "features": {
            "technical": {
                "bb_window": 20,
                "bb_num_std": 2.0,
                "rsi_window": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
            },
        },
        "risk": {"default_position_size_shares": 500},
    }
    return ChartService(config)


# -- get_price_history ---------------------------------------------


class TestGetPriceHistory:
    """Tests for ChartService.get_price_history."""

    def test_returns_dict_with_required_keys(self, chart_svc: ChartService) -> None:
        """Price history must include all expected keys."""
        result = chart_svc.get_price_history("SHEL.L", "3M")
        assert isinstance(result, dict)
        expected_keys = {
            "ticker", "period", "dates", "open", "high", "low", "close",
            "volume", "bb_upper", "bb_middle", "bb_lower",
            "benchmark_dates", "benchmark_close", "trade_entries",
        }
        assert expected_keys.issubset(result.keys())

    def test_close_prices_are_plain_floats(self, chart_svc: ChartService) -> None:
        """Close values must be plain floats, not numpy or pandas types."""
        result = chart_svc.get_price_history("SHEL.L", "1M")
        if result["close"]:
            assert isinstance(result["close"][0], float)

    def test_period_trims_data(self, chart_svc: ChartService) -> None:
        """1M period should return fewer points than 1Y."""
        short = chart_svc.get_price_history("SHEL.L", "1M")
        long = chart_svc.get_price_history("SHEL.L", "1Y")
        # 1M ~ 21 days, 1Y ~ 252 days
        assert len(short["dates"]) <= len(long["dates"])


# -- get_rsi_series ------------------------------------------------


class TestGetRsiSeries:
    """Tests for ChartService.get_rsi_series."""

    def test_rsi_bounded_0_100(self, chart_svc: ChartService) -> None:
        """RSI values must be in [0, 100] or None."""
        result = chart_svc.get_rsi_series("SHEL.L", "3M", window=14)
        for val in result["rsi"]:
            if val is not None:
                assert 0.0 <= val <= 100.0, f"RSI out of bounds: {val}"

    def test_rsi_returns_correct_window(self, chart_svc: ChartService) -> None:
        """The returned window value must match the requested one."""
        result = chart_svc.get_rsi_series("SHEL.L", "1Y", window=21)
        assert result["window"] == 21


# -- get_macd_series -----------------------------------------------


class TestGetMacdSeries:
    """Tests for ChartService.get_macd_series."""

    def test_histogram_colours_only_green_or_red(
        self, chart_svc: ChartService,
    ) -> None:
        """Histogram colours must be either 'green' or 'red'."""
        result = chart_svc.get_macd_series("SHEL.L", "1Y")
        for colour in result["histogram_colours"]:
            assert colour in ("green", "red"), f"Unexpected colour: {colour}"

    def test_macd_has_all_series(self, chart_svc: ChartService) -> None:
        """MACD result must have line, signal, histogram, and colours."""
        result = chart_svc.get_macd_series("SHEL.L", "3M")
        assert "macd_line" in result
        assert "macd_signal" in result
        assert "macd_histogram" in result
        assert "histogram_colours" in result


# -- get_ml_confidence_series --------------------------------------


class TestGetMlConfidenceSeries:
    """Tests for ChartService.get_ml_confidence_series."""

    def test_confidence_bounded_0_1(self, chart_svc: ChartService) -> None:
        """Confidence scores must be in [0.0, 1.0]."""
        result = chart_svc.get_ml_confidence_series("SHEL.L", "3M")
        for val in result["confidence"]:
            assert 0.0 <= val <= 1.0, f"Confidence out of bounds: {val}"


# -- get_forensic_analysis -----------------------------------------


class TestGetForensicAnalysis:
    """Tests for ChartService.get_forensic_analysis."""

    def test_forensic_returns_position_size_from_config(
        self, chart_svc: ChartService,
    ) -> None:
        """Position size must come from config (500 in test fixture)."""
        result = chart_svc.get_forensic_analysis("SHEL.L", "2024-06-15", 100.0)
        assert result["position_size"] == 500
