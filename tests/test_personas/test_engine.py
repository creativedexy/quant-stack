"""Tests for PersonaEngine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.personas.engine import PersonaEngine
from src.personas.persona import PersonaState, PersonaType


class TestPersonaEngineInit:
    """Initialisation and configuration tests."""

    def test_init_from_config(self, sample_config: dict, tmp_path: Path) -> None:
        sample_config["general"]["data_dir"] = str(tmp_path)
        engine = PersonaEngine(sample_config)

        assert len(engine.list_personas()) == 3
        assert "core" in engine.list_personas()
        assert "crisis" in engine.list_personas()
        assert "crypto_winter" in engine.list_personas()

    def test_always_on_starts_active(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        sample_config["general"]["data_dir"] = str(tmp_path)
        engine = PersonaEngine(sample_config)

        core = engine.get_persona("core")
        assert core is not None
        assert core.state == PersonaState.ACTIVE

    def test_event_triggered_starts_dormant(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        sample_config["general"]["data_dir"] = str(tmp_path)
        engine = PersonaEngine(sample_config)

        crisis = engine.get_persona("crisis")
        assert crisis is not None
        assert crisis.state == PersonaState.DORMANT

    def test_capital_validation(self, tmp_path: Path) -> None:
        config = {
            "general": {"data_dir": str(tmp_path)},
            "personas": {
                "enabled": True,
                "definitions": {
                    "a": {
                        "type": "always_on",
                        "capital_pct": 0.60,
                        "strategy": "momentum",
                        "universe": ["SPY"],
                        "risk": {},
                    },
                    "b": {
                        "type": "always_on",
                        "capital_pct": 0.50,
                        "strategy": "momentum",
                        "universe": ["QQQ"],
                        "risk": {},
                    },
                },
            },
        }
        with pytest.raises(ValueError, match="exceeding 100%"):
            PersonaEngine(config)


class TestPersonaEngineLifecycle:
    """State transition tests."""

    @pytest.fixture()
    def engine(self, sample_config: dict, tmp_path: Path) -> PersonaEngine:
        sample_config["general"]["data_dir"] = str(tmp_path)
        return PersonaEngine(sample_config)

    def test_activate_from_dormant(self, engine: PersonaEngine) -> None:
        assert engine.activate("crisis") is True
        assert engine.get_persona("crisis").state == PersonaState.ACTIVE
        assert engine.get_persona("crisis").activated_at is not None

    def test_activate_from_pending(self, engine: PersonaEngine) -> None:
        # Manually set to pending first
        engine._personas["crisis"].state = PersonaState.PENDING
        assert engine.activate("crisis") is True
        assert engine.get_persona("crisis").state == PersonaState.ACTIVE

    def test_activate_already_active(self, engine: PersonaEngine) -> None:
        # Core is already active
        assert engine.activate("core") is False

    def test_activate_unknown(self, engine: PersonaEngine) -> None:
        assert engine.activate("nonexistent") is False

    def test_deactivate(self, engine: PersonaEngine) -> None:
        assert engine.deactivate("core") is True
        assert engine.get_persona("core").state == PersonaState.DORMANT
        assert engine.get_persona("core").deactivated_at is not None

    def test_deactivate_already_dormant(self, engine: PersonaEngine) -> None:
        assert engine.deactivate("crisis") is True

    def test_pause(self, engine: PersonaEngine) -> None:
        assert engine.pause("core") is True
        assert engine.get_persona("core").state == PersonaState.PAUSED

    def test_pause_non_active(self, engine: PersonaEngine) -> None:
        assert engine.pause("crisis") is False  # dormant, can't pause

    def test_get_status(self, engine: PersonaEngine) -> None:
        status = engine.get_status()
        assert "core" in status
        assert status["core"]["state"] == "active"
        assert status["crisis"]["state"] == "dormant"


class TestPersonaEngineEvents:
    """Event detection tests."""

    @pytest.fixture()
    def engine(self, sample_config: dict, tmp_path: Path) -> PersonaEngine:
        sample_config["general"]["data_dir"] = str(tmp_path)
        return PersonaEngine(sample_config)

    def test_check_events_triggers_pending(
        self, engine: PersonaEngine, drawdown_ohlcv: pd.DataFrame
    ) -> None:
        market_data = {"VUSA.L": drawdown_ohlcv}
        newly_pending = engine.check_events(market_data)

        assert len(newly_pending) >= 1
        crisis_entry = next(
            (e for e in newly_pending if e["persona_id"] == "crisis"), None
        )
        assert crisis_entry is not None
        assert engine.get_persona("crisis").state == PersonaState.PENDING

    def test_check_events_no_trigger(
        self, engine: PersonaEngine, sample_ohlcv: pd.DataFrame
    ) -> None:
        # Use normal data with high drawdown threshold
        engine._personas["crisis"].activation_config["params"][
            "drawdown_threshold"
        ] = 0.90
        market_data = {"VUSA.L": sample_ohlcv}
        newly_pending = engine.check_events(market_data)

        assert len(newly_pending) == 0
        assert engine.get_persona("crisis").state == PersonaState.DORMANT

    def test_check_events_skips_active(
        self, engine: PersonaEngine, drawdown_ohlcv: pd.DataFrame
    ) -> None:
        engine._personas["crisis"].state = PersonaState.ACTIVE
        market_data = {"VUSA.L": drawdown_ohlcv}
        newly_pending = engine.check_events(market_data)

        # Crisis is already active, should not re-trigger
        assert not any(
            e["persona_id"] == "crisis" for e in newly_pending
        )


class TestPersonaEnginePersistence:
    """State persistence tests."""

    def test_persist_and_restore(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        sample_config["general"]["data_dir"] = str(tmp_path)

        # Create engine, activate crisis, persist
        engine1 = PersonaEngine(sample_config)
        engine1.activate("crisis")

        # Create a new engine — should restore state
        engine2 = PersonaEngine(sample_config)
        assert engine2.get_persona("crisis").state == PersonaState.ACTIVE

    def test_persist_and_restore_with_brief(
        self, sample_config: dict, tmp_path: Path
    ) -> None:
        sample_config["general"]["data_dir"] = str(tmp_path)
        engine = PersonaEngine(sample_config)

        # Generate a brief (with empty market data — will use template fallback)
        brief = engine.generate_brief("core", total_capital=100_000)
        assert brief is not None

        # Check brief was persisted
        latest = engine.get_latest_brief("core")
        assert latest is not None
        assert latest["persona_id"] == "core"
