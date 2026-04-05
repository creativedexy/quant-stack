"""Tests for DCAService.

Covers purchase recording, entry quality classification,
DCA summary calculations, and recommended limit logic.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.services.dca_service import (
    QUALITY_COLOURS,
    DCAService,
    _classify_entry_quality,
)
from src.services.dca_storage import DCAStorage


# -- Fixtures ------------------------------------------------------


@pytest.fixture
def dca_dir(tmp_path: Path) -> Path:
    """Temporary DCA data directory."""
    d = tmp_path / "data" / "dca"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def dca_storage(dca_dir: Path) -> DCAStorage:
    """SQLite-backed DCA storage in temp directory."""
    return DCAStorage(dca_dir)


@pytest.fixture
def dca_config(tmp_path: Path) -> dict:
    """Config dict pointing at temporary data directory."""
    return {
        "general": {
            "data_dir": str(tmp_path / "data"),
        },
    }


@pytest.fixture
def mock_chart_service() -> MagicMock:
    """Mock ChartService with RSI data."""
    svc = MagicMock()
    svc.get_rsi_series.return_value = {
        "dates": [
            "2024-01-15",
            "2024-02-15",
            "2024-03-15",
            "2024-04-15",
            "2024-05-15",
        ],
        "rsi": [30.0, 40.0, 55.0, 65.0, 25.0],
        "window": 14,
    }
    svc.get_price_history.return_value = {
        "dates": [
            "2024-01-15",
            "2024-02-15",
            "2024-03-15",
            "2024-04-15",
            "2024-05-15",
        ],
        "close": [2.50, 2.60, 2.70, 2.80, 2.90],
    }
    return svc


@pytest.fixture
def mock_data_service() -> MagicMock:
    """Mock DataService with price data."""
    import pandas as pd
    import numpy as np

    svc = MagicMock()
    idx = pd.date_range("2023-06-01", periods=252, freq="B")
    prices = pd.DataFrame(
        {"TEST.L": np.linspace(2.0, 3.0, 252)},
        index=idx,
    )
    svc.get_prices.return_value = prices
    return svc


@pytest.fixture
def mock_market_data_service() -> MagicMock:
    """Mock MarketDataService returning current price in pence."""
    svc = MagicMock()
    svc.get_intraday_prices.return_value = {"current": 280.0}
    return svc


@pytest.fixture
def dca_service(
    dca_config: dict,
    dca_storage: DCAStorage,
    mock_chart_service: MagicMock,
    mock_data_service: MagicMock,
    mock_market_data_service: MagicMock,
) -> DCAService:
    """DCAService wired to mocks with SQLite storage."""
    return DCAService(
        config=dca_config,
        chart_service=mock_chart_service,
        data_service=mock_data_service,
        market_data_service=mock_market_data_service,
        storage=dca_storage,
    )


# -- Tests ---------------------------------------------------------


class TestAddPurchase:
    """Tests for add_purchase method."""

    def test_add_purchase_persists(
        self, dca_service: DCAService, dca_storage: DCAStorage,
    ) -> None:
        """Adding a purchase persists to SQLite storage."""
        dca_service.add_purchase(
            "TEST.L", "2024-01-15", price=250.0, amount_gbp=1000.0,
        )
        data = dca_storage.get_all("TEST.L")
        assert len(data) == 1
        assert data[0]["date"] == "2024-01-15"

    def test_add_purchase_computes_shares_correctly(
        self, dca_service: DCAService,
    ) -> None:
        """GBP 1000 at 200p = 500 shares."""
        record = dca_service.add_purchase(
            "TEST.L", "2024-01-15", price=200.0, amount_gbp=1000.0,
        )
        # shares = floor(1000 / (200 / 100)) = floor(1000 / 2.0) = 500
        assert record["shares"] == 500

    def test_add_purchase_raises_on_duplicate_date(
        self, dca_service: DCAService,
    ) -> None:
        """Duplicate purchase date raises ValueError."""
        dca_service.add_purchase(
            "TEST.L", "2024-01-15", price=250.0, amount_gbp=1000.0,
        )
        with pytest.raises(ValueError, match="already recorded"):
            dca_service.add_purchase(
                "TEST.L", "2024-01-15", price=260.0, amount_gbp=500.0,
            )


class TestEntryQuality:
    """Tests for entry quality classification."""

    def test_entry_quality_below_35_is_best(self) -> None:
        """RSI below 35 should classify as 'best'."""
        assert _classify_entry_quality(30.0) == "best"
        assert _classify_entry_quality(34.9) == "best"
        assert _classify_entry_quality(10.0) == "best"

    def test_entry_quality_above_60_is_late(self) -> None:
        """RSI above 60 should classify as 'late'."""
        assert _classify_entry_quality(60.0) == "neutral"  # boundary
        assert _classify_entry_quality(60.1) == "late"
        assert _classify_entry_quality(80.0) == "late"


class TestDCASummary:
    """Tests for get_dca_summary method."""

    def test_avg_entry_price_weighted_by_shares(
        self, dca_service: DCAService,
    ) -> None:
        """Average entry price is weighted by shares, not simple average."""
        # Purchase 1: 500 shares at 200p
        dca_service.add_purchase(
            "TEST.L", "2024-01-15", price=200.0, amount_gbp=1000.0,
        )
        # Purchase 2: 250 shares at 400p
        dca_service.add_purchase(
            "TEST.L", "2024-02-15", price=400.0, amount_gbp=1000.0,
        )
        summary = dca_service.get_dca_summary("TEST.L")
        # Weighted avg = (200 * 500 + 400 * 250) / (500 + 250)
        #              = (100000 + 100000) / 750 = 266.67
        expected = round((200.0 * 500 + 400.0 * 250) / 750, 2)
        assert summary["avg_entry_price"] == expected

    def test_total_return_calculation(
        self, dca_service: DCAService,
    ) -> None:
        """Total return is based on current value vs total invested."""
        # Purchase: 500 shares at 200p, current price 280p
        dca_service.add_purchase(
            "TEST.L", "2024-01-15", price=200.0, amount_gbp=1000.0,
        )
        summary = dca_service.get_dca_summary("TEST.L")
        # Total invested = 1000 GBP
        assert summary["total_invested"] == 1000.0
        # Current value = 500 shares * (280p / 100) = 500 * 2.80 = 1400 GBP
        assert summary["current_value"] == 1400.0
        # Return = (1400 - 1000) / 1000 = 0.40
        assert summary["total_return_pct"] == 0.4
        assert summary["total_return_gbp"] == 400.0


class TestRecommendedLimit:
    """Tests for get_recommended_limit method."""

    def test_recommended_limit_never_above_52w_high_times_095(
        self, dca_service: DCAService,
    ) -> None:
        """Recommended limit <= 52w_high * 0.95."""
        dca_service.add_purchase(
            "TEST.L", "2024-01-15", price=200.0, amount_gbp=1000.0,
        )
        result = dca_service.get_recommended_limit("TEST.L")
        high_52w = dca_service._get_52w_high("TEST.L")
        assert result["recommended_limit_pence"] <= high_52w * 0.95 + 0.01

    def test_recommended_limit_never_above_avg_cost_times_108(
        self, dca_service: DCAService,
    ) -> None:
        """Recommended limit <= avg_cost_basis * 1.08."""
        dca_service.add_purchase(
            "TEST.L", "2024-01-15", price=200.0, amount_gbp=1000.0,
        )
        result = dca_service.get_recommended_limit("TEST.L")
        summary = dca_service.get_dca_summary("TEST.L")
        avg_cost = summary["avg_entry_price"]
        assert result["recommended_limit_pence"] <= avg_cost * 1.08 + 0.01


class TestChartData:
    """Tests for compute_dca_chart_data method."""

    def test_compute_dca_chart_data_has_one_dot_per_purchase(
        self, dca_service: DCAService,
    ) -> None:
        """Each purchase in the visible date range produces exactly one dot."""
        dca_service.add_purchase(
            "TEST.L", "2024-01-15", price=250.0, amount_gbp=1000.0,
        )
        dca_service.add_purchase(
            "TEST.L", "2024-03-15", price=270.0, amount_gbp=500.0,
        )
        chart = dca_service.compute_dca_chart_data("TEST.L")
        assert len(chart["purchase_dots"]) == 2

    def test_purchase_dot_colour_matches_quality_mapping(
        self, dca_service: DCAService,
    ) -> None:
        """Dot colour corresponds to the quality label."""
        # RSI on 2024-01-15 is 30.0 -> "best" -> #22c55e
        dca_service.add_purchase(
            "TEST.L", "2024-01-15", price=250.0, amount_gbp=1000.0,
        )
        # RSI on 2024-04-15 is 65.0 -> "late" -> #ef4444
        dca_service.add_purchase(
            "TEST.L", "2024-04-15", price=280.0, amount_gbp=500.0,
        )
        chart = dca_service.compute_dca_chart_data("TEST.L")
        dots = chart["purchase_dots"]
        dot_map = {d["date"]: d for d in dots}

        assert dot_map["2024-01-15"]["colour"] == QUALITY_COLOURS["best"]
        assert dot_map["2024-04-15"]["colour"] == QUALITY_COLOURS["late"]
