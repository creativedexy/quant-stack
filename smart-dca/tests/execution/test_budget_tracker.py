"""Tests for BudgetTracker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.execution.budget_tracker import BudgetTracker
from src.utils.exceptions import BudgetError


@pytest.fixture
def assets_config() -> dict:
    return {
        "assets": {
            "BTCGBP": {
                "weekly_allocation_gbp": 10.0,
                "min_order_gbp": 5.0,
                "enabled": True,
                "funding_proxy": "BTCUSDT",
            },
            "ETHGBP": {
                "weekly_allocation_gbp": 10.0,
                "min_order_gbp": 5.0,
                "enabled": True,
                "funding_proxy": "ETHUSDT",
            },
            "XRPGBP": {
                "weekly_allocation_gbp": 5.0,
                "min_order_gbp": 2.0,
                "enabled": True,
                "funding_proxy": None,
            },
        }
    }


@pytest.fixture
def config(sample_settings: dict) -> dict:
    return sample_settings


@pytest.fixture
def tracker(
    config: dict, assets_config: dict, tmp_path: Path
) -> BudgetTracker:
    state_file = tmp_path / "budget_state.json"
    return BudgetTracker(config, assets_config, state_file)


class TestBudgetTracker:
    def test_fresh_state_full_allocation_remaining(
        self, tracker: BudgetTracker
    ) -> None:
        """New state file -> remaining == weekly_allocation."""
        assert tracker.get_remaining("BTCGBP") == 10.0
        assert tracker.get_remaining("XRPGBP") == 5.0

    def test_record_spend_reduces_remaining(
        self, tracker: BudgetTracker
    ) -> None:
        """Spend 5.0 of 10.0 -> remaining == 5.0."""
        tracker.record_spend("BTCGBP", 5.0)
        assert tracker.get_remaining("BTCGBP") == pytest.approx(5.0)

    def test_cannot_overspend(self, tracker: BudgetTracker) -> None:
        """Spend 6.0 then try 5.0 -> BudgetError."""
        tracker.record_spend("BTCGBP", 6.0)
        with pytest.raises(BudgetError):
            tracker.record_spend("BTCGBP", 5.0)

    def test_can_spend_false_below_min_order(
        self, tracker: BudgetTracker
    ) -> None:
        """Remaining 10.0, amount 1.0, min_order 5.0 -> False."""
        assert tracker.can_spend("BTCGBP", 1.0) is False

    def test_week_reset_after_boundary(
        self,
        config: dict,
        assets_config: dict,
        tmp_path: Path,
    ) -> None:
        """Mock datetime past Monday 09:00 -> spent resets to 0."""
        state_file = tmp_path / "budget_state.json"
        tracker = BudgetTracker(config, assets_config, state_file)
        tracker.record_spend("BTCGBP", 8.0)
        assert tracker.get_remaining("BTCGBP") == pytest.approx(2.0)

        # Advance past next Monday 09:00 UTC
        future = datetime.now(timezone.utc) + timedelta(weeks=1, hours=1)
        remaining = tracker.get_remaining("BTCGBP", now=future)
        assert remaining == pytest.approx(10.0)

    def test_state_persists_across_instances(
        self,
        config: dict,
        assets_config: dict,
        tmp_path: Path,
    ) -> None:
        """Record spend, create new BudgetTracker -> remaining reflects spend."""
        state_file = tmp_path / "budget_state.json"
        tracker1 = BudgetTracker(config, assets_config, state_file)
        tracker1.record_spend("BTCGBP", 7.0)

        tracker2 = BudgetTracker(config, assets_config, state_file)
        assert tracker2.get_remaining("BTCGBP") == pytest.approx(3.0)

    def test_corrupt_state_file_reinitialises(
        self,
        config: dict,
        assets_config: dict,
        tmp_path: Path,
    ) -> None:
        """Write garbage to state file -> no crash, fresh state."""
        state_file = tmp_path / "budget_state.json"
        state_file.write_text("{{not json}}", encoding="utf-8")

        tracker = BudgetTracker(config, assets_config, state_file)
        assert tracker.get_remaining("BTCGBP") == pytest.approx(10.0)

    def test_all_assets_present_after_init(
        self, tracker: BudgetTracker
    ) -> None:
        """Three assets in config -> all three in get_all_states()."""
        states = tracker.get_all_states()
        assert set(states.keys()) == {"BTCGBP", "ETHGBP", "XRPGBP"}

    def test_record_spend_raises_budget_error_with_details(
        self, tracker: BudgetTracker
    ) -> None:
        """Error message includes asset, requested, and remaining."""
        tracker.record_spend("BTCGBP", 8.0)
        with pytest.raises(BudgetError) as exc_info:
            tracker.record_spend("BTCGBP", 5.0)
        err = exc_info.value
        assert err.asset == "BTCGBP"
        assert err.requested == 5.0
        assert err.remaining == pytest.approx(2.0)

    def test_week_start_is_monday_utc(
        self, tracker: BudgetTracker
    ) -> None:
        """After reset, week_start_utc.weekday() == 0 (Monday)."""
        tracker.reset_week("BTCGBP")
        state = tracker.get_all_states()["BTCGBP"]
        week_start = datetime.fromisoformat(state.week_start_utc)
        assert week_start.weekday() == 0
        assert week_start.tzinfo is not None
