"""Tests for src/deployment/memory.py."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from src.deployment.memory import MemoryManager


@pytest.fixture
def memory_dir(tmp_path) -> Path:
    """Temporary directory for memory files."""
    return tmp_path / "deployment"


@pytest.fixture
def cycles_dir(memory_dir) -> Path:
    """Temporary directory for cycle archives."""
    return memory_dir / "cycles"


@pytest.fixture
def memory(memory_dir, cycles_dir) -> MemoryManager:
    """MemoryManager pointed at temporary directories."""
    return MemoryManager(
        memory_path=memory_dir / "memory.json",
        cycles_dir=cycles_dir,
    )


# ------------------------------------------------------------------
# Core load / save
# ------------------------------------------------------------------


class TestLoadSave:
    """Tests for basic memory load and save operations."""

    def test_fresh_creation_when_no_file(self, memory, memory_dir):
        """Creates fresh state when memory.json does not exist."""
        state = memory.load()
        assert state["version"] == "1.0.0"
        assert state["last_cycle_id"] is None
        assert (memory_dir / "memory.json").exists()

    def test_load_save_round_trip(self, memory):
        """Save and reload preserves all fields."""
        state = memory.load()
        state["capital_available_gbp"] = 500.0
        state["universe_state"]["included_tickers"] = ["NVDA", "SMH"]
        memory.save(state)

        reloaded = memory.load()
        assert reloaded["capital_available_gbp"] == 500.0
        assert reloaded["universe_state"]["included_tickers"] == [
            "NVDA",
            "SMH",
        ]

    def test_atomic_write_no_partial(self, memory, memory_dir):
        """Atomic write produces valid JSON (no partial writes)."""
        state = memory.load()
        state["capital_available_gbp"] = 999.99
        memory.save(state)

        # Read raw file and verify it is valid JSON.
        raw = (memory_dir / "memory.json").read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert parsed["capital_available_gbp"] == 999.99

        # No .tmp file should remain.
        assert not (memory_dir / "memory.tmp").exists()


# ------------------------------------------------------------------
# Cycle management
# ------------------------------------------------------------------


class TestCycleManagement:
    """Tests for cycle start, end, and archival."""

    def test_start_cycle_creates_id(self, memory):
        """start_cycle returns an incrementing cycle ID."""
        c1 = memory.start_cycle()
        assert c1 == "cycle_001"

        c2 = memory.start_cycle()
        assert c2 == "cycle_002"

    def test_start_cycle_updates_state(self, memory):
        """start_cycle updates last_cycle_id and timestamp in state."""
        cycle_id = memory.start_cycle()
        state = memory.load()
        assert state["last_cycle_id"] == cycle_id
        assert state["last_cycle_timestamp"] is not None

    def test_end_cycle_archives(self, memory, cycles_dir):
        """end_cycle writes a cycle archive file."""
        cycle_id = memory.start_cycle()
        results = {
            "candidates_scanned": 15,
            "candidates_scored": 10,
            "candidates_above_threshold": 5,
            "trades_queued": 3,
            "errors": [],
        }
        memory.end_cycle(cycle_id, results)

        archive_path = cycles_dir / f"{cycle_id}.json"
        assert archive_path.exists()

        archived = json.loads(archive_path.read_text(encoding="utf-8"))
        assert archived["candidates_scanned"] == 15

    def test_end_cycle_appends_to_history(self, memory):
        """end_cycle adds a summary entry to evaluation_history."""
        cycle_id = memory.start_cycle()
        memory.end_cycle(cycle_id, {"candidates_scanned": 10})

        state = memory.load()
        assert len(state["evaluation_history"]) == 1
        assert state["evaluation_history"][0]["cycle_id"] == cycle_id


# ------------------------------------------------------------------
# Error logging
# ------------------------------------------------------------------


class TestErrorLogging:
    """Tests for pipeline error logging."""

    def test_log_error_persists(self, memory):
        """Logged errors are persisted in the error_log."""
        memory.load()  # Initialise state.
        memory.log_error(
            step="SCAN",
            error="yfinance timeout",
            fix_attempted="Retry with backoff",
            attempt=1,
        )

        state = memory.load()
        assert len(state["error_log"]) == 1
        assert state["error_log"][0]["step"] == "SCAN"


# ------------------------------------------------------------------
# Rejection / cooldown
# ------------------------------------------------------------------


class TestRejectionCooldown:
    """Tests for candidate rejection and cooldown management."""

    def test_log_rejection_adds_entry(self, memory):
        """log_rejection adds a candidate to the rejected list."""
        memory.load()
        memory.log_rejection("TSLA", "Too volatile", cooldown_days=7)

        state = memory.load()
        rejected = state["universe_state"]["rejected_candidates"]
        assert len(rejected) == 1
        assert rejected[0]["ticker"] == "TSLA"

    def test_get_rejected_tickers_filters_expired(self, memory):
        """get_rejected_tickers only returns tickers still in cooldown."""
        memory.load()

        # Add one with future cooldown.
        memory.log_rejection("TSLA", "volatile", cooldown_days=7)

        # Add one with expired cooldown (hack: set cooldown in the past).
        state = memory.load()
        state["universe_state"]["rejected_candidates"].append({
            "ticker": "GME",
            "rejected_at": "2020-01-01T00:00:00+00:00",
            "reason": "meme stock",
            "cooldown_until": "2020-01-08T00:00:00+00:00",
        })
        memory.save(state)

        active = memory.get_rejected_tickers()
        assert "TSLA" in active
        assert "GME" not in active

    def test_log_rejection_replaces_existing(self, memory):
        """Re-rejecting a ticker replaces the old entry."""
        memory.load()
        memory.log_rejection("TSLA", "reason 1", cooldown_days=7)
        memory.log_rejection("TSLA", "reason 2", cooldown_days=14)

        state = memory.load()
        rejected = state["universe_state"]["rejected_candidates"]
        tsla_entries = [r for r in rejected if r["ticker"] == "TSLA"]
        assert len(tsla_entries) == 1
        assert tsla_entries[0]["reason"] == "reason 2"


# ------------------------------------------------------------------
# Learnings
# ------------------------------------------------------------------


class TestLearnings:
    """Tests for learning summary retrieval."""

    def test_get_learnings_empty(self, memory):
        """Returns empty summary when no cycles have run."""
        memory.load()
        learnings = memory.get_learnings()
        assert learnings["total_cycles"] == 0
        assert learnings["recent_cycles"] == []

    def test_get_learnings_with_history(self, memory):
        """Returns recent cycle summaries up to max_cycles."""
        for i in range(7):
            cid = memory.start_cycle()
            memory.end_cycle(cid, {"candidates_scanned": i * 10})

        learnings = memory.get_learnings(max_cycles=5)
        assert len(learnings["recent_cycles"]) == 5
        assert learnings["total_cycles"] == 7


# ------------------------------------------------------------------
# Opus cost tracking
# ------------------------------------------------------------------


class TestCostTracking:
    """Tests for Opus API cost tracking."""

    def test_log_and_summarise_usage(self, memory):
        """Cost summary accumulates across multiple calls."""
        memory.load()
        memory.log_opus_usage("cycle_001", 1000, 500, "claude-opus-4-20250514")
        memory.log_opus_usage("cycle_001", 2000, 1000, "claude-opus-4-20250514")

        summary = memory.get_cost_summary()
        assert summary["total_tokens_in"] == 3000
        assert summary["total_tokens_out"] == 1500
        assert summary["num_calls"] == 2
        assert summary["estimated_cost_usd"] > 0
        assert summary["estimated_cost_gbp"] > 0


# ------------------------------------------------------------------
# Learning from cycles
# ------------------------------------------------------------------


class TestLearnFromCycle:
    """Tests for the learn_from_cycle method."""

    def test_high_scoring_rejection_goes_to_watchlist(self, memory):
        """Candidates scored >0.70 that were rejected go to watchlist."""
        memory.load()
        cycle_results = {
            "scored_candidates": [
                {"ticker": "NVDA", "composite_score": 0.85, "sector": "Tech"},
            ],
            "rejected_tickers": ["NVDA"],
        }
        memory.learn_from_cycle(cycle_results)

        state = memory.load()
        watchlist = state["universe_state"]["watchlist"]
        assert any(w["ticker"] == "NVDA" for w in watchlist)

    def test_low_scoring_gets_short_term_exclusion(self, memory):
        """Candidates scoring <0.40 get auto-excluded for 7 days."""
        memory.load()
        cycle_results = {
            "scored_candidates": [
                {"ticker": "BAD", "composite_score": 0.25, "sector": "Energy"},
            ],
            "rejected_tickers": [],
        }
        memory.learn_from_cycle(cycle_results)

        state = memory.load()
        rejected = state["universe_state"]["rejected_candidates"]
        assert any(r["ticker"] == "BAD" for r in rejected)

    def test_sector_trends_tracked(self, memory):
        """Sector scores are accumulated across learn_from_cycle calls."""
        memory.load()
        cycle_results = {
            "scored_candidates": [
                {"ticker": "A", "composite_score": 0.70, "sector": "Tech"},
                {"ticker": "B", "composite_score": 0.30, "sector": "Energy"},
            ],
            "rejected_tickers": [],
        }
        memory.learn_from_cycle(cycle_results)

        state = memory.load()
        assert "Tech" in state["sector_trends"]
        assert len(state["sector_trends"]["Tech"]) == 1


# ------------------------------------------------------------------
# Universe adjustments
# ------------------------------------------------------------------


class TestUniverseAdjustments:
    """Tests for scanner hints based on accumulated learning."""

    def test_boost_and_deprioritise_sectors(self, memory):
        """Sectors with consistent high/low scores are flagged."""
        memory.load()

        # Simulate 3 cycles with consistent sector performance.
        for _ in range(3):
            memory.learn_from_cycle({
                "scored_candidates": [
                    {"ticker": "X", "composite_score": 0.80, "sector": "Tech"},
                    {"ticker": "Y", "composite_score": 0.30, "sector": "Energy"},
                ],
                "rejected_tickers": [],
            })

        adjustments = memory.get_universe_adjustments()
        assert "Tech" in adjustments["boost_sectors"]
        assert "Energy" in adjustments["deprioritise_sectors"]


# ------------------------------------------------------------------
# Compaction
# ------------------------------------------------------------------


class TestCompaction:
    """Tests for memory compaction."""

    def test_compact_preserves_recent(self, memory):
        """Compaction keeps the most recent N cycles."""
        memory.load()

        for i in range(25):
            cid = memory.start_cycle()
            memory.end_cycle(cid, {"candidates_scanned": i})

        memory.compact(keep_cycles=10)

        state = memory.load()
        # 10 recent + 1 compacted summary entry.
        assert len(state["evaluation_history"]) == 11
        assert state["evaluation_history"][0]["type"] == "compacted_summary"


# ------------------------------------------------------------------
# Cycle diffing
# ------------------------------------------------------------------


class TestCycleDiffing:
    """Tests for diff_cycles method."""

    def test_diff_shows_new_and_removed(self, memory, cycles_dir):
        """Diff identifies new and removed candidates between cycles."""
        memory.load()

        # Write two fake cycle archives.
        cycle_a = {"scored_candidates": [
            {"ticker": "NVDA", "composite_score": 0.80},
            {"ticker": "AAPL", "composite_score": 0.60},
        ]}
        cycle_b = {"scored_candidates": [
            {"ticker": "NVDA", "composite_score": 0.85},
            {"ticker": "SMH", "composite_score": 0.70},
        ]}

        (cycles_dir / "cycle_001.json").write_text(
            json.dumps(cycle_a), encoding="utf-8"
        )
        (cycles_dir / "cycle_002.json").write_text(
            json.dumps(cycle_b), encoding="utf-8"
        )

        diff = memory.diff_cycles("cycle_001", "cycle_002")
        assert "SMH" in diff["new_candidates"]
        assert "AAPL" in diff["removed_candidates"]
        assert "NVDA" in diff["score_changes"]
        assert diff["score_changes"]["NVDA"]["delta"] == pytest.approx(
            0.05, abs=0.001
        )


# ------------------------------------------------------------------
# Backup
# ------------------------------------------------------------------


class TestBackup:
    """Tests for memory backup."""

    def test_backup_creates_file(self, memory, memory_dir):
        """backup() creates a timestamped copy of memory.json."""
        memory.load()  # Ensure file exists.
        backup_path = memory.backup()
        assert backup_path.exists()
        assert "backup_" in backup_path.name


# ------------------------------------------------------------------
# Thread safety
# ------------------------------------------------------------------


class TestThreadSafety:
    """Tests for concurrent memory access."""

    def test_concurrent_writes(self, memory):
        """Multiple threads writing simultaneously do not corrupt state."""
        memory.load()
        errors = []

        def writer(n: int) -> None:
            try:
                for _ in range(10):
                    memory.log_error(
                        step=f"STEP_{n}",
                        error="test",
                        fix_attempted="none",
                        attempt=1,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        state = memory.load()
        assert len(state["error_log"]) == 50  # 5 threads * 10 writes
