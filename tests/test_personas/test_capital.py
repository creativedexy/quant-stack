"""Tests for PersonaCapitalTracker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.personas.capital import PersonaCapitalTracker


class TestPersonaCapitalTracker:
    """Capital allocation and sizing tests."""

    def test_allocate_capital(self, sample_config: dict) -> None:
        tracker = PersonaCapitalTracker(sample_config)
        allocations = tracker.allocate_capital(100_000.0)

        assert allocations["core"] == 80_000.0
        assert allocations["crisis"] == 10_000.0
        assert allocations["crypto_winter"] == 5_000.0

    def test_allocate_capital_exceeds_100pct(self) -> None:
        config = {
            "general": {"data_dir": "data"},
            "personas": {
                "definitions": {
                    "a": {"capital_pct": 0.60},
                    "b": {"capital_pct": 0.50},
                },
            },
        }
        tracker = PersonaCapitalTracker(config)
        with pytest.raises(ValueError, match="exceeds 100%"):
            tracker.allocate_capital(100_000.0)

    def test_get_persona_budget(self, sample_config: dict) -> None:
        tracker = PersonaCapitalTracker(sample_config)
        tracker.allocate_capital(100_000.0)

        assert tracker.get_persona_budget("core") == 80_000.0
        assert tracker.get_persona_budget("nonexistent") == 0.0

    def test_get_unallocated(self, sample_config: dict) -> None:
        tracker = PersonaCapitalTracker(sample_config)
        tracker.allocate_capital(100_000.0)

        # 80% + 10% + 5% = 95%, so 5% unallocated
        assert tracker.get_unallocated() == pytest.approx(5_000.0, abs=0.01)

    def test_size_recommendation(self, sample_config: dict) -> None:
        tracker = PersonaCapitalTracker(sample_config)
        tracker.allocate_capital(100_000.0)

        # Core has 80k, max_weight=0.25 -> max 20k for one position
        # At price=55.0, that's 363 shares
        qty = tracker.size_recommendation("core", "VUSA.L", 55.0, max_weight=0.25)
        assert qty == 363

    def test_size_recommendation_zero_price(self, sample_config: dict) -> None:
        tracker = PersonaCapitalTracker(sample_config)
        tracker.allocate_capital(100_000.0)

        qty = tracker.size_recommendation("core", "X", 0.0, max_weight=0.25)
        assert qty == 0

    def test_size_recommendation_zero_budget(self, sample_config: dict) -> None:
        tracker = PersonaCapitalTracker(sample_config)
        # Don't allocate capital — budget is 0

        qty = tracker.size_recommendation("core", "VUSA.L", 55.0)
        assert qty == 0

    def test_get_summary(self, sample_config: dict) -> None:
        tracker = PersonaCapitalTracker(sample_config)
        tracker.allocate_capital(100_000.0)

        summary = tracker.get_summary()
        assert summary["total_capital"] == 100_000.0
        assert summary["persona_count"] == 3
        assert summary["unallocated"] == pytest.approx(5_000.0, abs=0.01)

    def test_persist_and_load(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        # Override data dir to use tmp
        config = {**sample_config}
        config["general"] = {"data_dir": str(tmp_path)}

        tracker = PersonaCapitalTracker(config)
        tracker.allocate_capital(50_000.0)
        tracker.persist()

        # Create a new tracker and load
        tracker2 = PersonaCapitalTracker(config)
        loaded = tracker2.load()

        assert loaded is True
        assert tracker2.get_persona_budget("core") == 40_000.0

    def test_load_no_file(self, sample_config: dict) -> None:
        tracker = PersonaCapitalTracker(sample_config)
        assert tracker.load() is False
