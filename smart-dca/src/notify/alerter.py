"""Alerter for Telegram and console notifications.

Sends notifications for: high score, buy executed, buy failed,
weekly summary. Falls back to console logging if Telegram
credentials are not configured.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.execution.budget_tracker import AllocationState
    from src.execution.buy_engine import BuyDecision
    from src.signals.scorer import CompositeScore

logger = get_logger(__name__)


class Alerter:
    """Sends notifications via Telegram or falls back to logging.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
    If not set, falls back to structlog INFO messages (never crashes).

    Args:
        config: Full settings dict from config_loader.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        scoring = config.get("scoring", {})
        self._notify_threshold = scoring.get("notify_threshold", 55)
        self._bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self._telegram_available = bool(self._bot_token and self._chat_id)

        if not self._telegram_available:
            logger.info("telegram_not_configured", fallback="console_only")

    def _send_telegram(self, message: str) -> None:
        """Send a message via Telegram bot API.

        Catches all exceptions — alerting must never block trading.

        Args:
            message: MarkdownV2 formatted message.
        """
        if not self._telegram_available:
            logger.info("alert_console_fallback", message=message)
            return

        try:
            import httpx

            url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            payload = {
                "chat_id": self._chat_id,
                "text": message,
                "parse_mode": "MarkdownV2",
            }
            response = httpx.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("telegram_sent", chat_id=self._chat_id)
        except Exception as exc:
            logger.error("telegram_send_failed", error=str(exc))

    def send_score_alert(self, score: CompositeScore, asset: str) -> None:
        """Send alert when score is building but below buy threshold.

        Only sends when score >= notify_threshold but below buy_threshold.

        Args:
            score: CompositeScore from signal scorer.
            asset: Trading pair symbol.
        """
        if score.score < self._notify_threshold:
            return

        message = (
            f"Signal building for {asset}\n"
            f"Score: {score.score:.0f}/100\n"
            f"Not buying yet - watching closely"
        )
        self._send_telegram(message)

    def send_buy_executed(self, decision: BuyDecision) -> None:
        """Send confirmation of successful buy.

        Args:
            decision: BuyDecision with executed=True.
        """
        order = decision.order_result
        price_str = f"{order.fill_price:,.2f}" if order else "N/A"
        amount_str = f"{order.amount_gbp:.2f}" if order else "N/A"

        lines = [
            f"BUY {decision.asset}",
            f"Amount: GBP {amount_str} at GBP {price_str}",
            f"Score: {decision.score.score:.0f}/100",
        ]

        for sig in decision.score.signals:
            contribution = sig.score * sig.weight * 100
            lines.append(
                f"  {sig.name:<16s} "
                f"{sig.score:.2f} x {sig.weight:.2f} = {contribution:.2f}"
            )

        message = "\n".join(lines)
        self._send_telegram(message)

    def send_buy_failed(self, decision: BuyDecision) -> None:
        """Send alert when score was high but execution failed.

        Args:
            decision: BuyDecision with executed=False.
        """
        message = (
            f"BUY FAILED for {decision.asset}\n"
            f"Score: {decision.score.score:.0f}/100\n"
            f"Reason: {decision.reason}"
        )
        self._send_telegram(message)

    def send_weekly_summary(
        self, states: dict[str, AllocationState]
    ) -> None:
        """Send weekly spend summary per asset.

        Args:
            states: Dict of asset -> AllocationState from BudgetTracker.
        """
        lines = ["Weekly Summary"]
        total_spent = 0.0
        total_allocation = 0.0

        for asset, state in states.items():
            remaining = state.weekly_allocation_gbp - state.spent_this_week_gbp
            lines.append(
                f"  {asset}: spent GBP {state.spent_this_week_gbp:.2f} "
                f"/ GBP {state.weekly_allocation_gbp:.2f} "
                f"(GBP {remaining:.2f} remaining)"
            )
            total_spent += state.spent_this_week_gbp
            total_allocation += state.weekly_allocation_gbp

        lines.append(
            f"Total: GBP {total_spent:.2f} / GBP {total_allocation:.2f}"
        )

        message = "\n".join(lines)
        self._send_telegram(message)
