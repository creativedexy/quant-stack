"""Persona analyst -- Claude API research brief generation.

Each persona has a ``mandate`` (system prompt) defining its analytical
perspective.  The analyst sends market data and strategy signals to
Claude, receiving a narrative research brief and trade recommendations.

Falls back to template-based output if ``ANTHROPIC_API_KEY`` is not set.

Usage::

    from src.personas.analyst import PersonaAnalyst
    analyst = PersonaAnalyst(config)
    brief = analyst.generate_brief(persona, market_snapshot, signals, trigger)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.personas.persona import (
    PersonaBrief,
    TradeRecommendation,
    TradingPersona,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PersonaAnalyst:
    """Generates research briefs using Claude API or template fallback.

    Args:
        config: Project configuration dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        personas_cfg = config.get("personas", {})
        self._model: str = personas_cfg.get("model", "claude-sonnet-4-6-20250514")
        self._has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))

        if self._has_api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
                logger.info("PersonaAnalyst initialised with Claude API")
            except ImportError:
                logger.warning(
                    "anthropic package not installed; "
                    "falling back to template briefs"
                )
                self._has_api_key = False
                self._client = None
        else:
            self._client = None
            logger.info(
                "No ANTHROPIC_API_KEY set; using template-based briefs"
            )

    def generate_brief(
        self,
        persona: TradingPersona,
        market_snapshot: dict[str, Any],
        signals: pd.DataFrame | None = None,
        recommendations: list[TradeRecommendation] | None = None,
        trigger_reason: str = "",
    ) -> PersonaBrief:
        """Generate a research brief for a persona.

        Args:
            persona: The trading persona requesting analysis.
            market_snapshot: Dictionary of market data (prices, indicators).
            signals: Strategy signal DataFrame (optional).
            recommendations: Pre-computed trade recommendations (optional).
            trigger_reason: What triggered this brief (for event personas).

        Returns:
            A complete :class:`PersonaBrief` with narrative and recommendations.
        """
        recs = recommendations or []

        if self._has_api_key and self._client is not None:
            narrative = self._call_claude(
                persona, market_snapshot, signals, recs, trigger_reason
            )
        else:
            narrative = self._template_brief(
                persona, market_snapshot, signals, recs, trigger_reason
            )

        return PersonaBrief(
            persona_id=persona.persona_id,
            timestamp=datetime.now(timezone.utc),
            narrative=narrative,
            recommendations=recs,
            market_data=market_snapshot,
            event_trigger=trigger_reason or None,
        )

    # ------------------------------------------------------------------
    # Claude API path
    # ------------------------------------------------------------------

    def _call_claude(
        self,
        persona: TradingPersona,
        market_snapshot: dict[str, Any],
        signals: pd.DataFrame | None,
        recommendations: list[TradeRecommendation],
        trigger_reason: str,
    ) -> str:
        """Call Claude API with the persona's mandate and market data."""
        system_prompt = self._build_system_prompt(persona)
        user_message = self._build_user_message(
            persona, market_snapshot, signals, recommendations, trigger_reason
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            narrative = response.content[0].text
            logger.info(
                "Claude brief generated for %s (%d chars)",
                persona.persona_id,
                len(narrative),
            )
            return narrative

        except Exception as exc:
            logger.error(
                "Claude API call failed for %s: %s",
                persona.persona_id,
                exc,
            )
            return self._template_brief(
                persona, market_snapshot, signals, recommendations,
                trigger_reason,
            )

    def _build_system_prompt(self, persona: TradingPersona) -> str:
        """Build the Claude system prompt from the persona's mandate."""
        return (
            f"{persona.mandate}\n\n"
            f"You are advising on a notional budget of "
            f"{persona.capital_pct:.0%} of total portfolio capital.\n"
            f"Your universe is: {', '.join(persona.universe)}.\n"
            f"Risk parameters: max drawdown {persona.risk_params.get('max_drawdown', 'N/A')}, "
            f"max position weight {persona.risk_params.get('max_weight', 'N/A')}.\n\n"
            "Provide:\n"
            "1. A concise market analysis from your mandate's perspective\n"
            "2. Specific actionable recommendations with entry reasoning\n"
            "3. Key risk factors to watch\n\n"
            "Use UK English. Be direct and specific."
        )

    def _build_user_message(
        self,
        persona: TradingPersona,
        market_snapshot: dict[str, Any],
        signals: pd.DataFrame | None,
        recommendations: list[TradeRecommendation],
        trigger_reason: str,
    ) -> str:
        """Build the user message with market data and context."""
        parts = []

        if trigger_reason:
            parts.append(f"## Event Trigger\n{trigger_reason}\n")

        parts.append("## Market Data")
        for ticker, data in market_snapshot.items():
            if isinstance(data, dict):
                parts.append(f"\n### {ticker}")
                for key, value in data.items():
                    if isinstance(value, float):
                        parts.append(f"- {key}: {value:.4f}")
                    else:
                        parts.append(f"- {key}: {value}")

        if signals is not None and not signals.empty:
            parts.append("\n## Strategy Signals")
            parts.append(signals.tail(5).to_string())

        if recommendations:
            parts.append("\n## Pre-computed Recommendations")
            for rec in recommendations:
                parts.append(
                    f"- {rec.direction} {rec.ticker}: "
                    f"{rec.suggested_quantity} shares @ {rec.entry_price:.2f} "
                    f"({rec.reasoning})"
                )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Template fallback
    # ------------------------------------------------------------------

    def _template_brief(
        self,
        persona: TradingPersona,
        market_snapshot: dict[str, Any],
        signals: pd.DataFrame | None,
        recommendations: list[TradeRecommendation],
        trigger_reason: str,
    ) -> str:
        """Generate a template-based brief when Claude API is unavailable."""
        parts = [f"# {persona.name} -- Research Brief\n"]

        if trigger_reason:
            parts.append(f"**Trigger:** {trigger_reason}\n")

        parts.append(f"**Strategy:** {persona.strategy_name}")
        parts.append(f"**Universe:** {', '.join(persona.universe)}")
        parts.append(
            f"**Budget:** {persona.capital_pct:.0%} of total capital\n"
        )

        # Market summary
        parts.append("## Market Summary\n")
        for ticker, data in market_snapshot.items():
            if isinstance(data, dict):
                price = data.get("close", data.get("Close", "N/A"))
                change = data.get("change_pct", "N/A")
                parts.append(f"- **{ticker}**: {price} ({change})")

        # Signals
        if signals is not None and not signals.empty:
            parts.append("\n## Strategy Signals\n")
            latest = signals.iloc[-1] if len(signals) > 0 else pd.Series()
            for col in signals.columns:
                val = latest.get(col, 0)
                direction = "LONG" if val > 0 else "SHORT" if val < 0 else "FLAT"
                parts.append(f"- {col}: {direction} (signal={val})")

        # Recommendations
        if recommendations:
            parts.append("\n## Trade Recommendations\n")
            for rec in recommendations:
                parts.append(
                    f"- **{rec.direction} {rec.ticker}**: "
                    f"{rec.suggested_quantity} shares @ "
                    f"{rec.entry_price:.2f} "
                    f"(~GBP {rec.suggested_size_gbp:,.0f})"
                )
                if rec.reasoning:
                    parts.append(f"  Reasoning: {rec.reasoning}")

        parts.append(
            "\n---\n*Template brief -- Claude API unavailable. "
            "Set ANTHROPIC_API_KEY for AI-powered analysis.*"
        )

        return "\n".join(parts)
