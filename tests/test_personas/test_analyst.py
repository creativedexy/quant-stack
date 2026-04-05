"""Tests for PersonaAnalyst (template fallback, no live Claude API calls)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.personas.analyst import PersonaAnalyst
from src.personas.persona import (
    PersonaBrief,
    PersonaState,
    PersonaType,
    TradeRecommendation,
    TradingPersona,
)


@pytest.fixture()
def analyst(sample_config: dict) -> PersonaAnalyst:
    """Create an analyst with no API key (template fallback)."""
    with patch.dict("os.environ", {}, clear=False):
        # Ensure ANTHROPIC_API_KEY is absent
        import os
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            a = PersonaAnalyst(sample_config)
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old
    return a


@pytest.fixture()
def core_persona() -> TradingPersona:
    """A sample always-on persona."""
    return TradingPersona(
        persona_id="core",
        name="Core Portfolio",
        persona_type=PersonaType.ALWAYS_ON,
        state=PersonaState.ACTIVE,
        mandate="You are the Core Portfolio analyst.",
        strategy_name="momentum",
        strategy_config={"sma_window": 50},
        universe=["VUSA.L", "HMWO.L"],
        capital_pct=0.80,
        risk_params={"max_drawdown": 0.15, "max_weight": 0.25},
    )


class TestPersonaAnalystTemplate:
    """Template fallback brief generation (no API key)."""

    def test_generates_brief(
        self, analyst: PersonaAnalyst, core_persona: TradingPersona
    ) -> None:
        market_snapshot = {
            "VUSA.L": {"close": 55.30, "change_pct": "+0.45%"},
            "HMWO.L": {"close": 42.10, "change_pct": "-0.22%"},
        }

        brief = analyst.generate_brief(
            core_persona,
            market_snapshot,
            trigger_reason="Scheduled daily brief",
        )

        assert isinstance(brief, PersonaBrief)
        assert brief.persona_id == "core"
        assert "Core Portfolio" in brief.narrative
        assert "Scheduled daily brief" in brief.narrative
        assert brief.event_trigger == "Scheduled daily brief"

    def test_generates_brief_with_recommendations(
        self, analyst: PersonaAnalyst, core_persona: TradingPersona
    ) -> None:
        recs = [
            TradeRecommendation(
                ticker="VUSA.L",
                direction="BUY",
                suggested_size_gbp=5000.0,
                suggested_quantity=90,
                entry_price=55.30,
                reasoning="Momentum signal positive",
            ),
        ]

        brief = analyst.generate_brief(
            core_persona,
            {"VUSA.L": {"close": 55.30}},
            recommendations=recs,
        )

        assert len(brief.recommendations) == 1
        assert brief.recommendations[0].ticker == "VUSA.L"
        assert "VUSA.L" in brief.narrative
        assert "BUY" in brief.narrative

    def test_generates_brief_with_signals(
        self, analyst: PersonaAnalyst, core_persona: TradingPersona
    ) -> None:
        signals = pd.DataFrame(
            {"signal": [0, 0, 1]},
            index=pd.bdate_range("2024-06-10", periods=3),
        )

        brief = analyst.generate_brief(
            core_persona,
            {"VUSA.L": {"close": 55.30}},
            signals=signals,
        )

        assert "Strategy Signals" in brief.narrative

    def test_template_fallback_note(
        self, analyst: PersonaAnalyst, core_persona: TradingPersona
    ) -> None:
        brief = analyst.generate_brief(core_persona, {})

        assert "Template brief" in brief.narrative
        assert "ANTHROPIC_API_KEY" in brief.narrative

    def test_empty_market_data(
        self, analyst: PersonaAnalyst, core_persona: TradingPersona
    ) -> None:
        brief = analyst.generate_brief(core_persona, {})

        assert isinstance(brief, PersonaBrief)
        assert brief.persona_id == "core"


class TestPersonaAnalystClaude:
    """Claude API integration (mocked)."""

    def test_calls_claude_api(
        self, sample_config: dict, core_persona: TradingPersona
    ) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="AI-generated analysis here.")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        # Create analyst with API key and inject mock client
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            analyst = PersonaAnalyst(sample_config)
        analyst._client = mock_client
        analyst._has_api_key = True

        brief = analyst.generate_brief(
            core_persona,
            {"VUSA.L": {"close": 55.30}},
            trigger_reason="Test trigger",
        )

        assert brief.narrative == "AI-generated analysis here."
        mock_client.messages.create.assert_called_once()

    def test_claude_api_failure_falls_back(
        self, sample_config: dict, core_persona: TradingPersona
    ) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            analyst = PersonaAnalyst(sample_config)
        analyst._client = mock_client
        analyst._has_api_key = True

        brief = analyst.generate_brief(core_persona, {})

        # Should fall back to template
        assert "Template brief" in brief.narrative
