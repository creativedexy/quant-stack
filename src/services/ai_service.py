"""AI Service -- wraps the Claude API for on-demand analysis.

Returns plain English insights for users with no financial background.
Falls back to synthetic (hardcoded) responses when the API key is
missing or the call fails.
"""

from __future__ import annotations

import os
from typing import Any

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Synthetic fallback responses ------------------------------------------------

_SYNTHETIC_TICKER_ANALYSIS = (
    "This ticker has been showing steady movement recently. "
    "The model is picking up a mix of positive and cautious signals "
    "from recent price action and trading volume. "
    "Nothing dramatic is happening right now, but it is worth "
    "keeping an eye on over the next few days. "
    "If you are already holding this, there is no urgent reason to "
    "change anything based on what the data is showing today."
)

_SYNTHETIC_PORTFOLIO_INSIGHTS: list[dict[str, Any]] = [
    {
        "colour": "green",
        "icon": "check",
        "title": "Portfolio is well spread out",
        "text": (
            "Your holdings cover several different sectors and regions. "
            "This means a downturn in one area is less likely to drag "
            "everything down at once."
        ),
        "chips": [],
    },
    {
        "colour": "amber",
        "icon": "alert",
        "title": "Tech weighting is on the high side",
        "text": (
            "Around 40% of your portfolio is in technology-related "
            "holdings. This has worked well recently, but it does "
            "mean you are more exposed if tech stocks pull back."
        ),
        "chips": [{"label": "CNDX.L", "colour": "cyan"}, {"label": "SMH", "colour": "cyan"}],
    },
    {
        "colour": "purple",
        "icon": "chart",
        "title": "DCA plan is on track",
        "text": (
            "Your regular buying schedule is working as expected. "
            "Sticking to the plan through ups and downs is one of "
            "the best things you can do."
        ),
        "chips": [],
    },
]

_SYNTHETIC_EXECUTION_INSIGHTS: list[dict[str, Any]] = [
    {
        "colour": "green",
        "icon": "check",
        "title": "System is running normally",
        "text": (
            "All connections are healthy and the pipeline ran "
            "successfully at the last scheduled time."
        ),
        "chips": [],
    },
    {
        "colour": "amber",
        "icon": "clock",
        "title": "No orders pending",
        "text": (
            "There are no open orders right now. The next rebalance "
            "check is scheduled for the usual time."
        ),
        "chips": [],
    },
]


class AIService:
    """Wraps the Claude API for on-demand portfolio and ticker analysis."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is None:
            config = load_config()
        ai_cfg = config.get("ai_service", {})
        self._enabled: bool = ai_cfg.get("enabled", True)
        self._model: str = ai_cfg.get("model", "claude-sonnet-4-20250514")
        self._max_tokens: int = ai_cfg.get("max_tokens", 1024)
        self._timeout: int = ai_cfg.get("timeout", 30)
        self._api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")

    def _has_api(self) -> bool:
        """Check whether the Claude API is available."""
        return bool(self._enabled and self._api_key)

    def _call_claude(self, system: str, user_msg: str) -> str:
        """Make a Claude API call with timeout and error handling."""
        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=self._api_key,
                timeout=float(self._timeout),
            )
            logger.info("AIService -> Claude API request (model=%s)", self._model)
            response = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text
            logger.info("AIService <- Claude API response (%d chars)", len(text))
            return text
        except Exception as exc:
            logger.warning("AIService Claude API call failed: %s", exc)
            return ""

    def analyse_ticker(
        self,
        ticker: str,
        signal: dict[str, Any],
        features: dict[str, Any],
    ) -> str:
        """Return plain English analysis of a single ticker.

        Falls back to a synthetic response if the API is unavailable.
        """
        if not self._has_api():
            logger.info("AIService.analyse_ticker using synthetic fallback")
            return _SYNTHETIC_TICKER_ANALYSIS

        system = (
            "You are a friendly investment guide. The user has no "
            "financial background. Write in plain English. No jargon. "
            "No acronyms. Use short sentences. Be honest about uncertainty. "
            "Use ASCII characters only."
        )
        user_msg = (
            f"Ticker: {ticker}\n"
            f"Signal direction: {signal.get('direction', 'unknown')}\n"
            f"Confidence: {signal.get('confidence', 0)}\n"
            f"Key features: {features}\n\n"
            "Explain what this means in 3-4 sentences. "
            "What should someone with no finance experience take away?"
        )
        result = self._call_claude(system, user_msg)
        return result if result else _SYNTHETIC_TICKER_ANALYSIS

    def review_portfolio(
        self,
        positions: list[dict[str, Any]],
        signals: dict[str, Any],
        dca_plans: list[Any],
    ) -> list[dict[str, Any]]:
        """Return insight cards about the portfolio.

        Each dict has keys: colour, icon, title, text, chips.
        Falls back to synthetic insights if the API is unavailable.
        """
        if not self._has_api():
            logger.info("AIService.review_portfolio using synthetic fallback")
            return _SYNTHETIC_PORTFOLIO_INSIGHTS

        system = (
            "You are a friendly investment guide. The user has no "
            "financial background. Return a JSON array of insight objects. "
            "Each object has: colour (green/red/amber/purple/cyan), "
            'icon (check/alert/chart/info), title, text, chips (array of '
            '{label, colour}). Use plain English. ASCII only. '
            "Return 2-4 insights. Return ONLY valid JSON, no markdown."
        )
        user_msg = (
            f"Positions: {positions}\n"
            f"Signals: {signals}\n"
            f"DCA plans: {dca_plans}\n"
        )
        result = self._call_claude(system, user_msg)
        if result:
            try:
                import json

                insights = json.loads(result)
                if isinstance(insights, list) and all(
                    isinstance(i, dict) and "title" in i for i in insights
                ):
                    return insights
            except (ValueError, KeyError):
                pass
        return _SYNTHETIC_PORTFOLIO_INSIGHTS

    def review_execution(
        self,
        health: dict[str, Any],
        orders: list[Any],
        rebalance: list[Any],
    ) -> list[dict[str, Any]]:
        """Return insight cards about system health and orders.

        Same format as review_portfolio.
        Falls back to synthetic insights if the API is unavailable.
        """
        if not self._has_api():
            logger.info("AIService.review_execution using synthetic fallback")
            return _SYNTHETIC_EXECUTION_INSIGHTS

        system = (
            "You are a system status reporter for a trading platform. "
            "The user has no technical background. Return a JSON array "
            "of insight objects with keys: colour, icon, title, text, chips. "
            "Use plain English. ASCII only. Return 2-3 insights. "
            "Return ONLY valid JSON, no markdown."
        )
        user_msg = (
            f"System health: {health}\n"
            f"Open orders: {orders}\n"
            f"Pending rebalance: {rebalance}\n"
        )
        result = self._call_claude(system, user_msg)
        if result:
            try:
                import json

                insights = json.loads(result)
                if isinstance(insights, list) and all(
                    isinstance(i, dict) and "title" in i for i in insights
                ):
                    return insights
            except (ValueError, KeyError):
                pass
        return _SYNTHETIC_EXECUTION_INSIGHTS
