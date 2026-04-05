"""Tests for persona data model."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.personas.persona import (
    PersonaBrief,
    PersonaState,
    PersonaType,
    TradeRecommendation,
    TradingPersona,
)


class TestPersonaState:
    """PersonaState enum tests."""

    def test_values(self) -> None:
        assert PersonaState.DORMANT.value == "dormant"
        assert PersonaState.PENDING.value == "pending"
        assert PersonaState.ACTIVE.value == "active"
        assert PersonaState.PAUSED.value == "paused"

    def test_from_string(self) -> None:
        assert PersonaState("active") == PersonaState.ACTIVE


class TestPersonaType:
    """PersonaType enum tests."""

    def test_values(self) -> None:
        assert PersonaType.ALWAYS_ON.value == "always_on"
        assert PersonaType.EVENT_TRIGGERED.value == "event_triggered"


class TestTradingPersona:
    """TradingPersona dataclass tests."""

    def test_from_config_always_on(self) -> None:
        defn = {
            "name": "Core Portfolio",
            "type": "always_on",
            "capital_pct": 0.80,
            "strategy": "momentum",
            "strategy_config": {"sma_window": 50},
            "universe": ["VUSA.L", "HMWO.L"],
            "risk": {"max_drawdown": 0.15},
            "mandate": "Test mandate",
        }
        persona = TradingPersona.from_config("core", defn)

        assert persona.persona_id == "core"
        assert persona.name == "Core Portfolio"
        assert persona.persona_type == PersonaType.ALWAYS_ON
        assert persona.state == PersonaState.ACTIVE  # always-on starts active
        assert persona.capital_pct == 0.80
        assert persona.strategy_name == "momentum"
        assert persona.universe == ["VUSA.L", "HMWO.L"]
        assert persona.mandate == "Test mandate"

    def test_from_config_event_triggered(self) -> None:
        defn = {
            "name": "Crisis Alpha",
            "type": "event_triggered",
            "capital_pct": 0.10,
            "strategy": "mean_reversion",
            "universe": ["VUSA.L"],
            "risk": {"max_drawdown": 0.25},
            "activation": {
                "detector": "drawdown",
                "params": {"drawdown_threshold": 0.10},
            },
        }
        persona = TradingPersona.from_config("crisis", defn)

        assert persona.persona_type == PersonaType.EVENT_TRIGGERED
        assert persona.state == PersonaState.DORMANT  # event starts dormant
        assert persona.activation_config["detector"] == "drawdown"

    def test_from_config_defaults(self) -> None:
        defn = {"capital_pct": 0.5, "universe": ["SPY"]}
        persona = TradingPersona.from_config("test", defn)

        assert persona.name == "Test"
        assert persona.strategy_name == "momentum"  # default
        assert persona.signal_method == "equal_weight_filtered"

    def test_to_dict(self) -> None:
        persona = TradingPersona(
            persona_id="test",
            name="Test",
            persona_type=PersonaType.ALWAYS_ON,
            state=PersonaState.ACTIVE,
            mandate="Test mandate",
            strategy_name="momentum",
            strategy_config={},
            universe=["SPY"],
            capital_pct=0.5,
            risk_params={"max_drawdown": 0.15},
            activated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        d = persona.to_dict()

        assert d["persona_id"] == "test"
        assert d["state"] == "active"
        assert d["persona_type"] == "always_on"
        assert d["activated_at"] == "2024-01-01T00:00:00+00:00"
        assert d["deactivated_at"] is None


class TestTradeRecommendation:
    """TradeRecommendation dataclass tests."""

    def test_to_dict(self) -> None:
        rec = TradeRecommendation(
            ticker="VUSA.L",
            direction="BUY",
            suggested_size_gbp=5000.0,
            suggested_quantity=90,
            entry_price=55.30,
            reasoning="RSI oversold at 22",
            risk_metrics={"rsi": 22},
        )
        d = rec.to_dict()

        assert d["ticker"] == "VUSA.L"
        assert d["direction"] == "BUY"
        assert d["suggested_size_gbp"] == 5000.0
        assert d["suggested_quantity"] == 90
        assert d["entry_price"] == 55.3
        assert d["risk_metrics"]["rsi"] == 22


class TestPersonaBrief:
    """PersonaBrief dataclass tests."""

    def test_to_dict(self) -> None:
        brief = PersonaBrief(
            persona_id="core",
            timestamp=datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc),
            narrative="Market looks strong.",
            recommendations=[
                TradeRecommendation(
                    ticker="VUSA.L",
                    direction="BUY",
                    suggested_size_gbp=5000.0,
                    suggested_quantity=90,
                    entry_price=55.30,
                ),
            ],
            market_data={"VUSA.L": {"close": 55.30}},
            event_trigger="Test trigger",
        )
        d = brief.to_dict()

        assert d["persona_id"] == "core"
        assert d["narrative"] == "Market looks strong."
        assert len(d["recommendations"]) == 1
        assert d["recommendations"][0]["ticker"] == "VUSA.L"
        assert d["event_trigger"] == "Test trigger"

    def test_to_dict_empty_recommendations(self) -> None:
        brief = PersonaBrief(
            persona_id="core",
            timestamp=datetime(2024, 6, 15, tzinfo=timezone.utc),
            narrative="Nothing to report.",
            recommendations=[],
        )
        d = brief.to_dict()
        assert d["recommendations"] == []
        assert d["event_trigger"] is None
