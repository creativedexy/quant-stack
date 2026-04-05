"""Budget tracker that enforces weekly allocation limits per asset.

Persists state to a JSON file so it survives process restarts.
Week boundary resets happen automatically on first access after the
configured reset point (default: Monday 09:00 UTC).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.utils.exceptions import BudgetError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AllocationState:
    """Per-asset budget state for the current week.

    Attributes:
        asset: Trading pair symbol.
        weekly_allocation_gbp: Maximum spend per week from config.
        spent_this_week_gbp: Amount already spent this week.
        week_start_utc: Start of the current budget week.
        last_updated: Timestamp of last state mutation.
    """

    asset: str
    weekly_allocation_gbp: float
    spent_this_week_gbp: float
    week_start_utc: str  # ISO format string for JSON serialisation
    last_updated: str  # ISO format string for JSON serialisation

    @property
    def week_start_dt(self) -> datetime:
        return datetime.fromisoformat(self.week_start_utc)

    @property
    def last_updated_dt(self) -> datetime:
        return datetime.fromisoformat(self.last_updated)


class BudgetTracker:
    """Enforces weekly GBP allocation limits per asset.

    State is persisted to a JSON file. Writes are atomic (write to
    temp file, then rename) to avoid corruption on crash.

    Args:
        config: Full settings dict from config_loader.
        assets_config: Assets dict from load_assets().
        state_file: Path to the JSON state file.
    """

    def __init__(
        self,
        config: dict,
        assets_config: dict,
        state_file: Path,
    ) -> None:
        self._config = config
        self._assets = assets_config.get("assets", {})
        self._state_file = state_file
        schedule = config.get("schedule", {})
        self._reset_hour = schedule.get("weekly_reset_hour_utc", 9)
        self._states: dict[str, AllocationState] = {}
        self._load_or_init()

    def _load_or_init(self) -> None:
        """Load state from disk or initialise fresh state."""
        if self._state_file.exists():
            try:
                raw = self._state_file.read_text(encoding="utf-8")
                data = json.loads(raw)
                for asset, state_dict in data.items():
                    self._states[asset] = AllocationState(**state_dict)
                logger.info("budget_state_loaded", path=str(self._state_file))
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.error(
                    "corrupt_state_file",
                    path=str(self._state_file),
                    error=str(exc),
                )
                self._states = {}

        # Ensure all configured assets have state entries
        for asset, cfg in self._assets.items():
            if asset not in self._states:
                now = datetime.now(timezone.utc)
                self._states[asset] = AllocationState(
                    asset=asset,
                    weekly_allocation_gbp=cfg.get("weekly_allocation_gbp", 0.0),
                    spent_this_week_gbp=0.0,
                    week_start_utc=self._compute_week_start(now).isoformat(),
                    last_updated=now.isoformat(),
                )
        self._persist()

    def _compute_week_start(self, dt: datetime) -> datetime:
        """Return the most recent Monday at reset_hour UTC on or before dt."""
        days_since_monday = dt.weekday()  # Monday=0
        monday = dt.replace(
            hour=self._reset_hour, minute=0, second=0, microsecond=0
        ) - timedelta(days=days_since_monday)
        # If we haven't passed the reset hour yet on Monday, go back a week
        if dt < monday:
            monday -= timedelta(weeks=1)
        return monday

    def _needs_reset(self, state: AllocationState, now: datetime) -> bool:
        """Return True if now is past the next weekly reset boundary."""
        next_reset = state.week_start_dt + timedelta(weeks=1)
        return now >= next_reset

    def get_remaining(self, asset: str, now: datetime | None = None) -> float:
        """Return GBP remaining for asset this week.

        Triggers automatic week reset if current time is past the
        weekly boundary.

        Args:
            asset: Trading pair symbol.
            now: Override current time (for testing).

        Returns:
            Remaining budget in GBP.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        state = self._states[asset]
        if self._needs_reset(state, now):
            self.reset_week(asset, now)
            state = self._states[asset]

        return state.weekly_allocation_gbp - state.spent_this_week_gbp

    def can_spend(self, asset: str, amount_gbp: float) -> bool:
        """Return True iff remaining >= amount and amount >= min_order.

        Args:
            asset: Trading pair symbol.
            amount_gbp: Proposed spend in GBP.

        Returns:
            True if the spend is allowed.
        """
        remaining = self.get_remaining(asset)
        min_order = self._assets.get(asset, {}).get("min_order_gbp", 0.0)
        return remaining >= amount_gbp and amount_gbp >= min_order

    def record_spend(self, asset: str, amount_gbp: float) -> None:
        """Record a spend against the asset's weekly budget.

        Persists state to disk immediately via atomic write.

        Args:
            asset: Trading pair symbol.
            amount_gbp: Amount spent in GBP.

        Raises:
            BudgetError: If amount would exceed remaining budget.
        """
        remaining = self.get_remaining(asset)
        if amount_gbp > remaining:
            raise BudgetError(
                asset=asset,
                requested=amount_gbp,
                remaining=remaining,
            )

        state = self._states[asset]
        state.spent_this_week_gbp += amount_gbp
        state.last_updated = datetime.now(timezone.utc).isoformat()
        self._persist()

        logger.info(
            "budget_spend_recorded",
            asset=asset,
            amount_gbp=amount_gbp,
            remaining=remaining - amount_gbp,
        )

    def reset_week(self, asset: str, now: datetime | None = None) -> None:
        """Reset the weekly budget for an asset.

        Args:
            asset: Trading pair symbol.
            now: Override current time (for testing).
        """
        if now is None:
            now = datetime.now(timezone.utc)

        state = self._states[asset]
        state.spent_this_week_gbp = 0.0
        state.week_start_utc = self._compute_week_start(now).isoformat()
        state.last_updated = now.isoformat()
        self._persist()

        logger.info("budget_week_reset", asset=asset, week_start=state.week_start_utc)

    def get_all_states(self) -> dict[str, AllocationState]:
        """Return current state for all configured assets."""
        return dict(self._states)

    def _persist(self) -> None:
        """Write state to disk atomically."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            asset: asdict(state) for asset, state in self._states.items()
        }
        # Atomic write: temp file then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._state_file.parent), suffix=".tmp"
        )
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            Path(tmp_path).replace(self._state_file)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
