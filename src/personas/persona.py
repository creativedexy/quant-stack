"""Core persona data model -- types, states, and dataclasses.

Defines the building blocks for trading personas: enums for lifecycle
states and persona types, plus dataclasses for personas, trade
recommendations, and research briefs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------

class PersonaState(str, Enum):
    """Lifecycle state of a trading persona."""

    DORMANT = "dormant"    # Event-triggered, conditions not met
    PENDING = "pending"    # Conditions detected, awaiting user confirmation
    ACTIVE = "active"      # Generating briefs and recommendations
    PAUSED = "paused"      # User paused -- no new analysis


class PersonaType(str, Enum):
    """Whether a persona runs continuously or waits for events."""

    ALWAYS_ON = "always_on"
    EVENT_TRIGGERED = "event_triggered"


# ------------------------------------------------------------------
# Core dataclass
# ------------------------------------------------------------------

@dataclass
class TradingPersona:
    """Runtime representation of a single trading persona.

    Attributes:
        persona_id: Unique key matching YAML config (e.g. ``"core"``).
        name: Human-readable display name.
        persona_type: Always-on or event-triggered.
        state: Current lifecycle state.
        mandate: System prompt for Claude API analysis.
        strategy_name: Key in the strategy registry.
        strategy_config: Passed to ``strategy_registry.create()``.
        universe: Ticker list for this persona.
        capital_pct: Notional budget as a fraction of total capital.
        risk_params: Max drawdown, max weight, etc.
        signal_method: Weight conversion method for sizing recommendations.
        activation_config: Detector type and params (event-triggered only).
        activated_at: When the persona was last activated.
        deactivated_at: When the persona was last deactivated.
        last_brief_at: When the last research brief was generated.
    """

    persona_id: str
    name: str
    persona_type: PersonaType
    state: PersonaState
    mandate: str
    strategy_name: str
    strategy_config: dict[str, Any]
    universe: list[str]
    capital_pct: float
    risk_params: dict[str, Any]
    signal_method: str = "equal_weight_filtered"
    activation_config: dict[str, Any] = field(default_factory=dict)
    activated_at: datetime | None = None
    deactivated_at: datetime | None = None
    last_brief_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary for JSON persistence."""
        return {
            "persona_id": self.persona_id,
            "name": self.name,
            "persona_type": self.persona_type.value,
            "state": self.state.value,
            "mandate": self.mandate,
            "strategy_name": self.strategy_name,
            "strategy_config": self.strategy_config,
            "universe": self.universe,
            "capital_pct": self.capital_pct,
            "risk_params": self.risk_params,
            "signal_method": self.signal_method,
            "activation_config": self.activation_config,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "deactivated_at": (
                self.deactivated_at.isoformat() if self.deactivated_at else None
            ),
            "last_brief_at": (
                self.last_brief_at.isoformat() if self.last_brief_at else None
            ),
        }

    @classmethod
    def from_config(
        cls, persona_id: str, definition: dict[str, Any]
    ) -> TradingPersona:
        """Create a persona from a YAML config definition block.

        Args:
            persona_id: Config key (e.g. ``"core"``).
            definition: The definition dict from ``personas.definitions``.

        Returns:
            A new ``TradingPersona`` in its initial state.
        """
        ptype = PersonaType(definition.get("type", "always_on"))
        initial_state = (
            PersonaState.ACTIVE
            if ptype == PersonaType.ALWAYS_ON
            else PersonaState.DORMANT
        )

        return cls(
            persona_id=persona_id,
            name=definition.get("name", persona_id.replace("_", " ").title()),
            persona_type=ptype,
            state=initial_state,
            mandate=definition.get("mandate", ""),
            strategy_name=definition.get("strategy", "momentum"),
            strategy_config=definition.get("strategy_config", {}),
            universe=definition.get("universe", []),
            capital_pct=definition.get("capital_pct", 0.0),
            risk_params=definition.get("risk", {}),
            signal_method=definition.get(
                "signal_method", "equal_weight_filtered"
            ),
            activation_config=definition.get("activation", {}),
        )


# ------------------------------------------------------------------
# Trade recommendation
# ------------------------------------------------------------------

@dataclass
class TradeRecommendation:
    """A single actionable trade suggestion from a persona.

    Attributes:
        ticker: Instrument identifier.
        direction: ``"BUY"`` or ``"SELL"``.
        suggested_size_gbp: Notional value from persona's budget.
        suggested_quantity: Number of shares/units.
        entry_price: Current market price at time of recommendation.
        reasoning: Human-readable explanation from strategy signals.
        risk_metrics: Supporting indicators (RSI, drawdown, etc.).
    """

    ticker: str
    direction: str
    suggested_size_gbp: float
    suggested_quantity: int
    entry_price: float
    reasoning: str = ""
    risk_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "suggested_size_gbp": round(self.suggested_size_gbp, 2),
            "suggested_quantity": self.suggested_quantity,
            "entry_price": round(self.entry_price, 4),
            "reasoning": self.reasoning,
            "risk_metrics": self.risk_metrics,
        }


# ------------------------------------------------------------------
# Research brief
# ------------------------------------------------------------------

@dataclass
class PersonaBrief:
    """A complete research output from a persona.

    Contains both the Claude API narrative analysis and the structured
    trade recommendations.

    Attributes:
        persona_id: Which persona generated this brief.
        timestamp: When the brief was generated (UTC).
        narrative: Claude API research analysis text.
        recommendations: Actionable trade suggestions.
        market_data: Snapshot of supporting data at generation time.
        event_trigger: What triggered this brief (for event personas).
    """

    persona_id: str
    timestamp: datetime
    narrative: str
    recommendations: list[TradeRecommendation]
    market_data: dict[str, Any] = field(default_factory=dict)
    event_trigger: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary for JSON persistence."""
        return {
            "persona_id": self.persona_id,
            "timestamp": self.timestamp.isoformat(),
            "narrative": self.narrative,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "market_data": self.market_data,
            "event_trigger": self.event_trigger,
        }
