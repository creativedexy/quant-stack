"""Persistent memory for the deployment pipeline.

Manages the rolling state file (memory.json) and per-cycle archives.
All writes are atomic (write to .tmp, then rename) to avoid corruption.

Thread-safe: uses a Lock for all read/write operations.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

_FRESH_MEMORY: dict[str, Any] = {
    "version": "1.0.0",
    "last_cycle_id": None,
    "last_cycle_timestamp": None,
    "capital_available_gbp": 0.0,
    "capital_deployed_gbp": 0.0,
    "positions_opened": [],
    "universe_state": {
        "included_tickers": [],
        "excluded_tickers_structural": [],
        "rejected_candidates": [],
        "watchlist": [],
    },
    "evaluation_history": [],
    "config_overrides": {},
    "error_log": [],
    "opus_usage": [],
    "sector_trends": {},
}


class MemoryManager:
    """Persistent state manager for the deployment pipeline.

    Args:
        memory_path: Path to the rolling memory.json file.
        cycles_dir: Directory for per-cycle archive snapshots.
    """

    # Default paths relative to project root.
    _DEFAULT_MEMORY = Path(__file__).parent.parent.parent / "data" / "deployment" / "memory.json"
    _DEFAULT_CYCLES = Path(__file__).parent.parent.parent / "data" / "deployment" / "cycles"

    def __init__(
        self,
        memory_path: str | Path | None = None,
        cycles_dir: str | Path | None = None,
    ) -> None:
        self._memory_path = Path(memory_path) if memory_path else self._DEFAULT_MEMORY
        self._cycles_dir = Path(cycles_dir) if cycles_dir else self._DEFAULT_CYCLES
        self._lock = threading.Lock()

        # Ensure directories exist.
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._cycles_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MemoryManager":
        """Create a MemoryManager using paths from the deployment config."""
        mem_cfg = config.get("memory", {})
        return cls(
            memory_path=mem_cfg.get("memory_path"),
            cycles_dir=mem_cfg.get("cycles_dir"),
        )

    # ------------------------------------------------------------------
    # Core load / save
    # ------------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """Load current memory state. Creates fresh state if file missing.

        Returns:
            Memory state dict.
        """
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> dict[str, Any]:
        """Load without acquiring the lock (caller must hold it)."""
        if not self._memory_path.exists():
            logger.info("No memory file found -- creating fresh state")
            state = json.loads(json.dumps(_FRESH_MEMORY))  # deep copy
            self._save_unlocked(state)
            return state

        with open(self._memory_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, state: dict[str, Any]) -> None:
        """Write memory state atomically (write .tmp, then rename).

        Args:
            state: Full memory state dict to persist.
        """
        with self._lock:
            self._save_unlocked(state)

    def _save_unlocked(self, state: dict[str, Any]) -> None:
        """Save without acquiring the lock (caller must hold it)."""
        tmp_path = self._memory_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        # Atomic rename (on same filesystem).
        os.replace(str(tmp_path), str(self._memory_path))

    # ------------------------------------------------------------------
    # Cycle management
    # ------------------------------------------------------------------

    def start_cycle(self, cycle_id: str | None = None) -> str:
        """Start a new deployment cycle.

        Args:
            cycle_id: Optional override. Auto-generated if None.

        Returns:
            The cycle ID.
        """
        with self._lock:
            state = self._load_unlocked()

            if cycle_id is None:
                # Compute next cycle number.
                last_id = state.get("last_cycle_id") or "cycle_000"
                try:
                    num = int(last_id.split("_")[1]) + 1
                except (IndexError, ValueError):
                    num = 1
                cycle_id = f"cycle_{num:03d}"
            now = datetime.now(timezone.utc).isoformat()

            state["last_cycle_id"] = cycle_id
            state["last_cycle_timestamp"] = now

            self._save_unlocked(state)
            logger.info("Started deployment cycle %s", cycle_id)
            return cycle_id

    def end_cycle(
        self,
        cycle_id: str,
        results: dict[str, Any],
    ) -> None:
        """Archive a completed cycle and update rolling state.

        Args:
            cycle_id: The cycle being completed.
            results: Full cycle results to archive.
        """
        with self._lock:
            # Write cycle archive.
            archive_path = self._cycles_dir / f"{cycle_id}.json"
            with open(archive_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)

            # Update rolling state.
            state = self._load_unlocked()
            state["evaluation_history"].append({
                "cycle_id": cycle_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "candidates_scanned": results.get("candidates_scanned", 0),
                    "candidates_scored": results.get("candidates_scored", 0),
                    "candidates_above_threshold": results.get(
                        "candidates_above_threshold", 0
                    ),
                    "trades_queued": results.get("trades_queued", 0),
                    "errors": len(results.get("errors", [])),
                },
            })
            self._save_unlocked(state)
            logger.info("Archived cycle %s", cycle_id)

    # ------------------------------------------------------------------
    # Error logging
    # ------------------------------------------------------------------

    def log_error(
        self,
        step: str,
        error: str,
        fix_attempted: str,
        attempt: int,
    ) -> None:
        """Log a pipeline error to memory.

        Args:
            step: Pipeline step where the error occurred.
            error: Error description.
            fix_attempted: What fix was attempted.
            attempt: Retry attempt number.
        """
        with self._lock:
            state = self._load_unlocked()
            state["error_log"].append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "step": step,
                "error": error,
                "fix_attempted": fix_attempted,
                "attempt": attempt,
            })
            self._save_unlocked(state)

    # ------------------------------------------------------------------
    # Rejection / cooldown management
    # ------------------------------------------------------------------

    def log_rejection(
        self,
        ticker: str,
        reason: str,
        cooldown_days: int = 7,
    ) -> None:
        """Record a rejected candidate with a cooldown period.

        Args:
            ticker: Rejected ticker symbol.
            reason: Why it was rejected.
            cooldown_days: Days before the ticker can be re-scanned.
        """
        with self._lock:
            state = self._load_unlocked()
            now = datetime.now(timezone.utc)
            cooldown_until = now + timedelta(days=cooldown_days)

            # Remove any existing entry for the same ticker.
            rejected = state["universe_state"]["rejected_candidates"]
            rejected = [r for r in rejected if r["ticker"] != ticker]

            rejected.append({
                "ticker": ticker,
                "rejected_at": now.isoformat(),
                "reason": reason,
                "cooldown_until": cooldown_until.isoformat(),
            })

            state["universe_state"]["rejected_candidates"] = rejected
            self._save_unlocked(state)
            logger.info(
                "Rejected %s (cooldown until %s): %s",
                ticker,
                cooldown_until.date(),
                reason,
            )

    def get_rejected_tickers(self) -> list[str]:
        """Return tickers currently in cooldown (not yet expired).

        Returns:
            List of ticker strings with active cooldowns.
        """
        with self._lock:
            state = self._load_unlocked()

        now = datetime.now(timezone.utc)
        active = []

        for entry in state["universe_state"].get("rejected_candidates", []):
            try:
                cooldown_str = entry.get("cooldown_until", "")
                cooldown_dt = datetime.fromisoformat(cooldown_str)
                if cooldown_dt > now:
                    active.append(entry["ticker"])
            except (ValueError, TypeError):
                continue

        return active

    # ------------------------------------------------------------------
    # Learnings
    # ------------------------------------------------------------------

    def get_learnings(self, max_cycles: int = 5) -> dict[str, Any]:
        """Return a summary of the last N cycles for Opus context.

        Args:
            max_cycles: Maximum number of recent cycles to include.

        Returns:
            Dict with cycle summaries, rejected tickers, and sector trends.
        """
        with self._lock:
            state = self._load_unlocked()

        history = state.get("evaluation_history", [])
        recent = history[-max_cycles:] if history else []

        return {
            "recent_cycles": recent,
            "rejected_tickers": [
                {
                    "ticker": r["ticker"],
                    "reason": r["reason"],
                }
                for r in state["universe_state"].get(
                    "rejected_candidates", []
                )
            ],
            "watchlist": state["universe_state"].get("watchlist", []),
            "sector_trends": state.get("sector_trends", {}),
            "total_cycles": len(history),
        }

    # ------------------------------------------------------------------
    # Opus cost tracking
    # ------------------------------------------------------------------

    def log_opus_usage(
        self,
        cycle_id: str,
        tokens_in: int,
        tokens_out: int,
        model: str,
    ) -> None:
        """Record Opus API token usage for cost tracking.

        Args:
            cycle_id: Which cycle this call belongs to.
            tokens_in: Input tokens consumed.
            tokens_out: Output tokens generated.
            model: Model ID used.
        """
        with self._lock:
            state = self._load_unlocked()
            state.setdefault("opus_usage", []).append({
                "cycle_id": cycle_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "model": model,
            })
            self._save_unlocked(state)

    def get_cost_summary(self) -> dict[str, Any]:
        """Return cumulative Opus API cost summary.

        Uses approximate pricing: input $0.015/1K tokens, output $0.075/1K tokens.

        Returns:
            Dict with total_tokens_in, total_tokens_out, estimated_cost_usd,
            estimated_cost_gbp, cost_per_cycle_avg, num_calls.
        """
        with self._lock:
            state = self._load_unlocked()

        usage = state.get("opus_usage", [])

        total_in = sum(u.get("tokens_in", 0) for u in usage)
        total_out = sum(u.get("tokens_out", 0) for u in usage)

        # Approximate pricing (USD).
        cost_usd = (total_in / 1000 * 0.015) + (total_out / 1000 * 0.075)
        cost_gbp = cost_usd / 1.27  # Approximate GBP conversion.

        num_calls = len(usage)
        cycles_with_usage = len({u.get("cycle_id") for u in usage})

        return {
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "estimated_cost_usd": round(cost_usd, 4),
            "estimated_cost_gbp": round(cost_gbp, 4),
            "cost_per_cycle_avg_gbp": (
                round(cost_gbp / cycles_with_usage, 4)
                if cycles_with_usage > 0
                else 0.0
            ),
            "num_calls": num_calls,
        }

    # ------------------------------------------------------------------
    # Learning from cycles
    # ------------------------------------------------------------------

    def learn_from_cycle(self, cycle_results: dict[str, Any]) -> None:
        """Analyse completed cycle and update universe state.

        - High-scoring rejections (>0.70) go to watchlist.
        - Low-scoring candidates (<0.40) get short-term exclusion.
        - Sector trends are tracked.

        Args:
            cycle_results: Full results from the completed cycle.
        """
        with self._lock:
            state = self._load_unlocked()

            scored = cycle_results.get("scored_candidates", [])
            rejected_tickers = set(
                cycle_results.get("rejected_tickers", [])
            )

            now = datetime.now(timezone.utc)
            cooldown_until = (
                now + timedelta(days=7)
            ).isoformat()

            watchlist = state["universe_state"].get("watchlist", [])
            watchlist_tickers = {w["ticker"] for w in watchlist}

            sector_counts: dict[str, list[float]] = state.get(
                "sector_trends", {}
            )

            for candidate in scored:
                ticker = candidate.get("ticker", "")
                composite = candidate.get("composite_score", 0.0)
                sector = candidate.get("sector", "Unknown")

                # Track sector performance.
                sector_counts.setdefault(sector, [])
                sector_counts[sector].append(composite)

                # High-scoring rejection -> watchlist.
                if (
                    ticker in rejected_tickers
                    and composite >= 0.70
                    and ticker not in watchlist_tickers
                ):
                    watchlist.append({
                        "ticker": ticker,
                        "added_at": now.isoformat(),
                        "composite_score": composite,
                        "reason": "High-scoring rejection",
                    })
                    watchlist_tickers.add(ticker)

                # Low-scoring -> short-term exclusion.
                if composite < 0.40:
                    existing = state["universe_state"][
                        "rejected_candidates"
                    ]
                    if not any(r["ticker"] == ticker for r in existing):
                        existing.append({
                            "ticker": ticker,
                            "rejected_at": now.isoformat(),
                            "reason": "low_score_auto",
                            "cooldown_until": cooldown_until,
                        })

            state["universe_state"]["watchlist"] = watchlist
            state["sector_trends"] = sector_counts
            self._save_unlocked(state)

    def get_universe_adjustments(self) -> dict[str, Any]:
        """Return scanner hints based on accumulated learning.

        Returns:
            Dict with boost_sectors, deprioritise_sectors,
            watchlist_tickers, exclude_tickers.
        """
        with self._lock:
            state = self._load_unlocked()

        sector_trends = state.get("sector_trends", {})

        boost: list[str] = []
        deprioritise: list[str] = []

        for sector, scores in sector_trends.items():
            if len(scores) >= 3:
                avg = sum(scores) / len(scores)
                if avg >= 0.65:
                    boost.append(sector)
                elif avg < 0.40:
                    deprioritise.append(sector)

        watchlist_tickers = [
            w["ticker"]
            for w in state["universe_state"].get("watchlist", [])
        ]

        return {
            "boost_sectors": boost,
            "deprioritise_sectors": deprioritise,
            "watchlist_tickers": watchlist_tickers,
            "exclude_tickers": self.get_rejected_tickers(),
        }

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def compact(self, keep_cycles: int = 20) -> None:
        """Archive old cycles to prevent memory bloat.

        Keeps the most recent ``keep_cycles`` in detail. Older entries
        in ``evaluation_history`` are summarised.

        Args:
            keep_cycles: Number of recent cycles to preserve in full.
        """
        with self._lock:
            state = self._load_unlocked()

            history = state.get("evaluation_history", [])
            if len(history) <= keep_cycles:
                return

            # Keep recent, summarise old.
            old = history[:-keep_cycles]
            recent = history[-keep_cycles:]

            summary = {
                "compacted_cycles": len(old),
                "compacted_at": datetime.now(timezone.utc).isoformat(),
                "total_scanned": sum(
                    e.get("summary", {}).get("candidates_scanned", 0)
                    for e in old
                ),
                "total_trades": sum(
                    e.get("summary", {}).get("trades_queued", 0)
                    for e in old
                ),
            }

            state["evaluation_history"] = [
                {"type": "compacted_summary", **summary},
                *recent,
            ]

            # Trim opus usage for old cycles.
            recent_ids = {e.get("cycle_id") for e in recent}
            state["opus_usage"] = [
                u
                for u in state.get("opus_usage", [])
                if u.get("cycle_id") in recent_ids
            ]

            self._save_unlocked(state)
            logger.info(
                "Compacted %d old cycles, kept %d recent",
                len(old),
                keep_cycles,
            )

    # ------------------------------------------------------------------
    # Cycle diffing
    # ------------------------------------------------------------------

    def diff_cycles(
        self,
        cycle_a: str,
        cycle_b: str,
    ) -> dict[str, Any]:
        """Show what changed between two cycles.

        Args:
            cycle_a: Earlier cycle ID.
            cycle_b: Later cycle ID.

        Returns:
            Dict with new_candidates, removed_candidates, score_changes.
        """
        a_path = self._cycles_dir / f"{cycle_a}.json"
        b_path = self._cycles_dir / f"{cycle_b}.json"

        a_data: dict[str, Any] = {}
        b_data: dict[str, Any] = {}

        if a_path.exists():
            with open(a_path, "r", encoding="utf-8") as f:
                a_data = json.load(f)

        if b_path.exists():
            with open(b_path, "r", encoding="utf-8") as f:
                b_data = json.load(f)

        a_tickers = {
            c.get("ticker")
            for c in a_data.get("scored_candidates", [])
        }
        b_tickers = {
            c.get("ticker")
            for c in b_data.get("scored_candidates", [])
        }

        new_candidates = list(b_tickers - a_tickers)
        removed_candidates = list(a_tickers - b_tickers)

        # Score changes for tickers in both.
        a_scores = {
            c["ticker"]: c.get("composite_score", 0.0)
            for c in a_data.get("scored_candidates", [])
        }
        b_scores = {
            c["ticker"]: c.get("composite_score", 0.0)
            for c in b_data.get("scored_candidates", [])
        }
        common = a_tickers & b_tickers
        score_changes = {
            t: {
                "before": a_scores.get(t, 0.0),
                "after": b_scores.get(t, 0.0),
                "delta": round(
                    b_scores.get(t, 0.0) - a_scores.get(t, 0.0), 4
                ),
            }
            for t in common
            if abs(b_scores.get(t, 0.0) - a_scores.get(t, 0.0)) > 0.001
        }

        return {
            "cycle_a": cycle_a,
            "cycle_b": cycle_b,
            "new_candidates": new_candidates,
            "removed_candidates": removed_candidates,
            "score_changes": score_changes,
        }

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------

    def backup(self) -> Path:
        """Create a backup of the current memory file.

        Returns:
            Path to the backup file.
        """
        if not self._memory_path.exists():
            return self._memory_path

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup_path = self._memory_path.with_suffix(
            f".backup_{timestamp}.json"
        )
        shutil.copy2(self._memory_path, backup_path)
        logger.info("Backed up memory to %s", backup_path)
        return backup_path
