"""Tests for the performance log module."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.monitor.performance_log import FillRecord, PerformanceLog


def _make_fill(
    asset: str = "BTCGBP",
    fill_price: float = 48000.0,
    amount_gbp: float = 10.0,
    quantity: float = 0.000208,
    naive_price: float = 48000.0,
    score_at_buy: float = 72.0,
    timestamp: str = "2024-03-04T14:30:00+00:00",
) -> FillRecord:
    """Helper to build a FillRecord with sensible defaults."""
    return FillRecord(
        asset=asset,
        fill_price=fill_price,
        amount_gbp=amount_gbp,
        quantity=quantity,
        naive_price=naive_price,
        score_at_buy=score_at_buy,
        timestamp=timestamp,
    )


class TestRecordAndRetrieve:
    """Test basic insert and retrieval."""

    def test_record_and_retrieve_fill(self, tmp_path: Path) -> None:
        """Record one fill then retrieve it via get_all_fills."""
        db = tmp_path / "perf.db"
        log = PerformanceLog(db)

        fill = _make_fill()
        log.record_fill(fill)

        fills = log.get_all_fills()
        assert len(fills) == 1
        assert fills[0].asset == "BTCGBP"
        assert fills[0].fill_price == 48000.0
        assert fills[0].amount_gbp == 10.0
        assert fills[0].quantity == 0.000208
        assert fills[0].naive_price == 48000.0
        assert fills[0].score_at_buy == 72.0
        assert fills[0].timestamp == "2024-03-04T14:30:00+00:00"


class TestOrdering:
    """Test timestamp ordering."""

    def test_fills_ordered_by_timestamp(self, tmp_path: Path) -> None:
        """Fills inserted out of chronological order are returned sorted."""
        db = tmp_path / "perf.db"
        log = PerformanceLog(db)

        # Insert in reverse chronological order
        log.record_fill(
            _make_fill(timestamp="2024-03-06T10:00:00+00:00")
        )
        log.record_fill(
            _make_fill(timestamp="2024-03-04T09:00:00+00:00")
        )
        log.record_fill(
            _make_fill(timestamp="2024-03-05T12:00:00+00:00")
        )

        fills = log.get_all_fills()
        assert len(fills) == 3
        assert fills[0].timestamp == "2024-03-04T09:00:00+00:00"
        assert fills[1].timestamp == "2024-03-05T12:00:00+00:00"
        assert fills[2].timestamp == "2024-03-06T10:00:00+00:00"


class TestAssetFilter:
    """Test filtering by asset."""

    def test_asset_filter(self, tmp_path: Path) -> None:
        """get_all_fills with asset filter returns only matching fills."""
        db = tmp_path / "perf.db"
        log = PerformanceLog(db)

        log.record_fill(_make_fill(asset="BTCGBP"))
        log.record_fill(_make_fill(asset="ETHGBP"))
        log.record_fill(_make_fill(asset="BTCGBP"))

        btc_fills = log.get_all_fills("BTCGBP")
        assert len(btc_fills) == 2
        assert all(f.asset == "BTCGBP" for f in btc_fills)

        eth_fills = log.get_all_fills("ETHGBP")
        assert len(eth_fills) == 1
        assert eth_fills[0].asset == "ETHGBP"


class TestImprovementPct:
    """Test improvement percentage calculations."""

    def test_improvement_pct_positive_when_beat_naive(
        self, tmp_path: Path
    ) -> None:
        """Fill price below naive price yields positive improvement."""
        db = tmp_path / "perf.db"
        log = PerformanceLog(db)

        log.record_fill(
            _make_fill(fill_price=47000.0, naive_price=48000.0)
        )

        summaries = log.compute_performance_summary()
        assert len(summaries) == 1
        assert summaries[0]["improvement_pct"] > 0
        # Exact value: ((48000 - 47000) / 48000) * 100 ~= 2.083%
        assert summaries[0]["improvement_pct"] == pytest.approx(
            2.0833333, rel=1e-3
        )

    def test_improvement_pct_negative_when_worse(
        self, tmp_path: Path
    ) -> None:
        """Fill price above naive price yields negative improvement."""
        db = tmp_path / "perf.db"
        log = PerformanceLog(db)

        log.record_fill(
            _make_fill(fill_price=49000.0, naive_price=48000.0)
        )

        summaries = log.compute_performance_summary()
        assert len(summaries) == 1
        assert summaries[0]["improvement_pct"] < 0
        # Exact value: ((48000 - 49000) / 48000) * 100 ~= -2.083%
        assert summaries[0]["improvement_pct"] == pytest.approx(
            -2.0833333, rel=1e-3
        )


class TestEmptyDb:
    """Test behaviour with no records."""

    def test_empty_db_returns_empty_list(self, tmp_path: Path) -> None:
        """An empty database returns an empty list from get_all_fills."""
        db = tmp_path / "perf.db"
        log = PerformanceLog(db)

        assert log.get_all_fills() == []

    def test_empty_db_summary_returns_empty_list(
        self, tmp_path: Path
    ) -> None:
        """An empty database returns an empty list from compute_performance_summary."""
        db = tmp_path / "perf.db"
        log = PerformanceLog(db)

        assert log.compute_performance_summary() == []

    def test_empty_db_weekly_returns_empty_list(
        self, tmp_path: Path
    ) -> None:
        """An empty database returns an empty list from get_weekly_performance."""
        db = tmp_path / "perf.db"
        log = PerformanceLog(db)

        assert log.get_weekly_performance() == []
