"""Tests for event detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.personas.detectors import (
    CryptoDrawdownDetector,
    DetectionResult,
    DrawdownDetector,
    ThresholdDetector,
    VIXSpikeDetector,
    detector_registry,
)


class TestDetectorRegistry:
    """Registry pattern tests."""

    def test_list_detectors(self) -> None:
        names = detector_registry.list_detectors()
        assert "drawdown" in names
        assert "crypto_drawdown" in names
        assert "vix_spike" in names
        assert "threshold" in names

    def test_create_detector(self) -> None:
        d = detector_registry.create(
            "drawdown", params={"drawdown_threshold": 0.10}
        )
        assert isinstance(d, DrawdownDetector)
        assert d.params["drawdown_threshold"] == 0.10

    def test_create_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown detector"):
            detector_registry.create("nonexistent")


class TestDrawdownDetector:
    """DrawdownDetector tests."""

    def test_triggers_on_drawdown(self, drawdown_ohlcv: pd.DataFrame) -> None:
        detector = DrawdownDetector(
            params={"drawdown_threshold": 0.10, "lookback_days": 30}
        )
        result = detector.check({"VUSA.L": drawdown_ohlcv})

        assert result.triggered is True
        assert "VUSA.L" in result.reason
        assert result.metrics["worst_drawdown"] >= 0.10

    def test_no_trigger_below_threshold(
        self, sample_ohlcv: pd.DataFrame
    ) -> None:
        detector = DrawdownDetector(
            params={"drawdown_threshold": 0.50, "lookback_days": 21}
        )
        result = detector.check({"SPY": sample_ohlcv})

        assert result.triggered is False

    def test_empty_data(self) -> None:
        detector = DrawdownDetector(params={"drawdown_threshold": 0.10})
        result = detector.check({"SPY": pd.DataFrame()})

        assert result.triggered is False

    def test_no_data_for_ticker(self) -> None:
        detector = DrawdownDetector(params={"drawdown_threshold": 0.10})
        result = detector.check({})

        assert result.triggered is False


class TestCryptoDrawdownDetector:
    """CryptoDrawdownDetector tests."""

    def test_triggers_on_crypto_drawdown(
        self, drawdown_ohlcv: pd.DataFrame
    ) -> None:
        detector = CryptoDrawdownDetector(
            params={
                "index_ticker": "BTC-USD",
                "drawdown_threshold": 0.10,
                "lookback_days": 30,
            }
        )
        result = detector.check({"BTC-USD": drawdown_ohlcv})

        assert result.triggered is True
        assert "BTC-USD" in result.reason
        assert result.metrics["current_drawdown"] >= 0.10

    def test_no_data(self) -> None:
        detector = CryptoDrawdownDetector(
            params={"index_ticker": "BTC-USD", "drawdown_threshold": 0.30}
        )
        result = detector.check({})

        assert result.triggered is False
        assert "No data" in result.reason


class TestVIXSpikeDetector:
    """VIXSpikeDetector tests."""

    def test_triggers_on_vix_spike_and_market_drop(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=10, freq="B")
        vix_data = pd.DataFrame(
            {"Close": [35.0] * 10}, index=dates
        )
        # Market drops from 100 to 90 (10% drop)
        market_data = pd.DataFrame(
            {
                "Close": np.linspace(100, 90, 10),
                "Open": np.linspace(100, 90, 10),
                "High": np.linspace(101, 91, 10),
                "Low": np.linspace(99, 89, 10),
            },
            index=dates,
        )

        detector = VIXSpikeDetector(
            params={
                "vix_ticker": "^VIX",
                "vix_threshold": 30,
                "market_drop_pct": 0.05,
                "lookback_days": 5,
            }
        )
        result = detector.check({"^VIX": vix_data, "SPY": market_data})

        assert result.triggered is True
        assert result.metrics["vix_level"] >= 30

    def test_no_trigger_vix_low(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=10, freq="B")
        vix_data = pd.DataFrame({"Close": [15.0] * 10}, index=dates)

        detector = VIXSpikeDetector(
            params={"vix_ticker": "^VIX", "vix_threshold": 30}
        )
        result = detector.check({"^VIX": vix_data})

        assert result.triggered is False

    def test_no_vix_data(self) -> None:
        detector = VIXSpikeDetector(params={"vix_ticker": "^VIX"})
        result = detector.check({})

        assert result.triggered is False


class TestThresholdDetector:
    """ThresholdDetector tests."""

    def test_triggers(self, drawdown_ohlcv: pd.DataFrame) -> None:
        detector = ThresholdDetector(
            params={
                "watch_ticker": "BTC-USD",
                "threshold": 0.10,
                "lookback_days": 30,
            }
        )
        result = detector.check({"BTC-USD": drawdown_ohlcv})

        assert result.triggered is True

    def test_no_watch_ticker(self) -> None:
        detector = ThresholdDetector(params={})
        result = detector.check({})

        assert result.triggered is False
        assert "No watch_ticker" in result.reason


class TestDetectionResult:
    """DetectionResult dataclass tests."""

    def test_defaults(self) -> None:
        r = DetectionResult(triggered=False)
        assert r.triggered is False
        assert r.reason == ""
        assert r.metrics == {}
