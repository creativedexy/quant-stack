"""Persona capital tracker -- virtual sub-accounting for notional budgets.

Tracks each persona's notional capital allocation for sizing trade
recommendations.  This is advisory-only: no real positions are held
by the tracker.

Usage::

    from src.personas.capital import PersonaCapitalTracker
    tracker = PersonaCapitalTracker(config)
    tracker.allocate_capital(100_000.0)
    budget = tracker.get_persona_budget("core")
    qty = tracker.size_recommendation("core", "VUSA.L", 55.30, max_weight=0.25)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PersonaCapitalTracker:
    """Virtual sub-accounting for persona notional budgets.

    Each persona gets a fixed percentage of total capital for sizing
    trade recommendations.  No real positions are tracked here -- this
    is purely for advisory output sizing.

    Args:
        config: Project configuration dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        personas_cfg = config.get("personas", {})
        self._definitions = personas_cfg.get("definitions", {})

        data_dir_name = config.get("general", {}).get("data_dir", "data")
        self._persist_path = (
            _PROJECT_ROOT / data_dir_name / "processed" / "personas" / "capital.json"
        )
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)

        # persona_id -> allocated capital value
        self._allocations: dict[str, float] = {}
        self._total_capital: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate_capital(self, total: float) -> dict[str, float]:
        """Distribute total capital across personas by their ``capital_pct``.

        Args:
            total: Total portfolio capital in base currency (GBP).

        Returns:
            Dictionary of persona_id -> allocated capital value.

        Raises:
            ValueError: If persona capital percentages sum to more than 1.0.
        """
        self._total_capital = total
        total_pct = sum(
            defn.get("capital_pct", 0.0)
            for defn in self._definitions.values()
        )
        if total_pct > 1.0 + 1e-9:
            raise ValueError(
                f"Persona capital_pct values sum to {total_pct:.2%}, "
                "which exceeds 100%."
            )

        self._allocations = {}
        for persona_id, defn in self._definitions.items():
            pct = defn.get("capital_pct", 0.0)
            self._allocations[persona_id] = round(total * pct, 2)

        logger.info(
            "Capital allocated across %d personas (total: %.2f)",
            len(self._allocations),
            total,
        )
        return dict(self._allocations)

    def get_persona_budget(self, persona_id: str) -> float:
        """Return the notional budget for a specific persona.

        Args:
            persona_id: Persona identifier.

        Returns:
            Allocated capital value in base currency.
        """
        return self._allocations.get(persona_id, 0.0)

    def get_all_allocations(self) -> dict[str, float]:
        """Return a copy of all persona allocations."""
        return dict(self._allocations)

    def get_unallocated(self) -> float:
        """Return capital not assigned to any persona."""
        allocated = sum(self._allocations.values())
        return max(0.0, self._total_capital - allocated)

    def size_recommendation(
        self,
        persona_id: str,
        ticker: str,
        price: float,
        max_weight: float = 1.0,
    ) -> int:
        """Compute suggested quantity for a trade recommendation.

        Args:
            persona_id: Persona making the recommendation.
            ticker: Instrument identifier (for logging).
            price: Current market price per share.
            max_weight: Maximum fraction of persona budget for one position.

        Returns:
            Suggested number of whole shares (always >= 0).
        """
        budget = self.get_persona_budget(persona_id)
        if budget <= 0 or price <= 0:
            return 0

        max_value = budget * max_weight
        quantity = int(max_value // price)
        logger.debug(
            "Sized %s for %s: %d shares @ %.2f (budget=%.2f, max_weight=%.0f%%)",
            ticker,
            persona_id,
            quantity,
            price,
            budget,
            max_weight * 100,
        )
        return max(0, quantity)

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of capital allocations for the UI."""
        return {
            "total_capital": self._total_capital,
            "allocations": dict(self._allocations),
            "unallocated": self.get_unallocated(),
            "persona_count": len(self._allocations),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self) -> None:
        """Save current allocations to disk."""
        data = {
            "total_capital": self._total_capital,
            "allocations": self._allocations,
        }
        self._persist_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        logger.debug("Capital state persisted to %s", self._persist_path)

    def load(self) -> bool:
        """Load allocations from disk if the file exists.

        Returns:
            ``True`` if state was loaded, ``False`` if no file found.
        """
        if not self._persist_path.exists():
            return False

        data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        self._total_capital = data.get("total_capital", 0.0)
        self._allocations = data.get("allocations", {})
        logger.info(
            "Loaded capital state: total=%.2f, %d personas",
            self._total_capital,
            len(self._allocations),
        )
        return True
