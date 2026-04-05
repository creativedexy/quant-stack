"""Persona service -- UI-agnostic interface for trading personas.

Sits between the web routes and the persona engine, following the same
service layer pattern as the other services in ``src/services/``.

Usage::

    from src.services.persona_service import PersonaService
    svc = PersonaService(config)
    personas = svc.get_all_personas()
    svc.activate_persona("crisis")
    brief = svc.generate_brief_now("crisis")
"""

from __future__ import annotations

from typing import Any

from src.personas.engine import PersonaEngine
from src.personas.persona import PersonaBrief
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PersonaService:
    """Service layer for trading persona operations.

    Args:
        config: Project configuration dict.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            from src.utils.config import load_config
            config = load_config()

        self._config = config
        personas_cfg = config.get("personas", {})
        self._enabled = personas_cfg.get("enabled", False)

        if self._enabled:
            self._engine = PersonaEngine(config)
        else:
            self._engine = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all_personas(self) -> list[dict[str, Any]]:
        """Return summary of all personas for the dashboard card grid.

        Returns:
            List of persona summary dicts, or empty list if disabled.
        """
        if not self._engine:
            return []
        status = self._engine.get_status()
        return list(status.values())

    def get_persona_detail(self, persona_id: str) -> dict[str, Any] | None:
        """Detailed view of a single persona including latest brief.

        Args:
            persona_id: Persona identifier.

        Returns:
            Detail dict, or ``None`` if not found or disabled.
        """
        if not self._engine:
            return None

        persona = self._engine.get_persona(persona_id)
        if persona is None:
            return None

        status = self._engine.get_status().get(persona_id, {})
        latest_brief = self._engine.get_latest_brief(persona_id)

        return {
            **status,
            "mandate": persona.mandate,
            "signal_method": persona.signal_method,
            "activation_config": persona.activation_config,
            "latest_brief": latest_brief,
        }

    def get_latest_brief(self, persona_id: str) -> dict[str, Any] | None:
        """Load the most recent brief for a persona.

        Args:
            persona_id: Persona identifier.

        Returns:
            Brief dict, or ``None`` if not found.
        """
        if not self._engine:
            return None
        return self._engine.get_latest_brief(persona_id)

    def get_capital_summary(self) -> dict[str, Any]:
        """Return capital allocation summary across all personas.

        Returns:
            Summary dict with total, allocations, unallocated.
        """
        if not self._engine:
            return {"total_capital": 0, "allocations": {}, "unallocated": 0}
        return self._engine._capital.get_summary()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def activate_persona(self, persona_id: str) -> dict[str, Any]:
        """Activate a persona (user confirmation).

        Args:
            persona_id: Persona to activate.

        Returns:
            Result dict with success status and message.
        """
        if not self._engine:
            return {"success": False, "message": "Personas not enabled"}

        ok = self._engine.activate(persona_id)
        if ok:
            return {
                "success": True,
                "message": f"Persona '{persona_id}' activated",
            }
        return {
            "success": False,
            "message": f"Cannot activate '{persona_id}' from current state",
        }

    def deactivate_persona(self, persona_id: str) -> dict[str, Any]:
        """Deactivate a persona.

        Args:
            persona_id: Persona to deactivate.

        Returns:
            Result dict.
        """
        if not self._engine:
            return {"success": False, "message": "Personas not enabled"}

        ok = self._engine.deactivate(persona_id)
        return {
            "success": ok,
            "message": (
                f"Persona '{persona_id}' deactivated" if ok
                else f"Cannot deactivate '{persona_id}'"
            ),
        }

    def pause_persona(self, persona_id: str) -> dict[str, Any]:
        """Pause an active persona.

        Args:
            persona_id: Persona to pause.

        Returns:
            Result dict.
        """
        if not self._engine:
            return {"success": False, "message": "Personas not enabled"}

        ok = self._engine.pause(persona_id)
        return {
            "success": ok,
            "message": (
                f"Persona '{persona_id}' paused" if ok
                else f"Cannot pause '{persona_id}'"
            ),
        }

    def generate_brief_now(
        self,
        persona_id: str,
        total_capital: float = 100_000.0,
    ) -> dict[str, Any] | None:
        """Manually trigger a research brief for a persona.

        Args:
            persona_id: Persona to generate a brief for.
            total_capital: Total portfolio capital for sizing.

        Returns:
            Brief dict, or ``None`` if generation failed.
        """
        if not self._engine:
            return None

        brief = self._engine.generate_brief(
            persona_id, total_capital=total_capital
        )
        if brief is None:
            return None
        return brief.to_dict()

    def check_events(self) -> list[dict[str, Any]]:
        """Run event detection for all dormant personas.

        Returns:
            List of newly pending persona dicts.
        """
        if not self._engine:
            return []
        # Event detection needs market data -- pass empty for now
        # (scheduler integration provides real data)
        return self._engine.check_events({})
