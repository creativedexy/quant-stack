"""Tests for Alerter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.notify.alerter import Alerter
from src.signals.scorer import CompositeScore


@pytest.fixture
def config(sample_settings: dict) -> dict:
    return sample_settings


@pytest.fixture
def alerter(config: dict) -> Alerter:
    """Alerter with no Telegram credentials (console fallback)."""
    return Alerter(config)


class TestAlerter:
    def test_no_credentials_no_crash(self, alerter: Alerter) -> None:
        """Initialise with no env vars -> no exception."""
        assert alerter._telegram_available is False

    def test_score_alert_sent_above_notify_threshold(
        self, alerter: Alerter
    ) -> None:
        """score 60, threshold 55 -> _send_telegram called."""
        score = CompositeScore(score=60.0, should_buy=False)
        with patch.object(alerter, "_send_telegram") as mock_send:
            alerter.send_score_alert(score, "BTCGBP")
            mock_send.assert_called_once()

    def test_score_alert_not_sent_below_threshold(
        self, alerter: Alerter
    ) -> None:
        """score 50, threshold 55 -> telegram not called."""
        score = CompositeScore(score=50.0, should_buy=False)
        with patch.object(alerter, "_send_telegram") as mock_send:
            alerter.send_score_alert(score, "BTCGBP")
            mock_send.assert_not_called()

    def test_buy_executed_message_contains_asset(
        self, alerter: Alerter
    ) -> None:
        """Verify asset name in message string."""
        from src.execution.buy_engine import BuyDecision
        from src.execution.order_manager import OrderResult

        order = OrderResult(
            order_id="12345",
            asset="BTCGBP",
            status="NEW",
            fill_price=50000.0,
            quantity=0.0002,
            amount_gbp=10.0,
            timestamp=datetime.now(timezone.utc),
            raw_response={},
        )
        decision = BuyDecision(
            asset="BTCGBP",
            executed=True,
            score=CompositeScore(score=80.0, should_buy=True),
            order_result=order,
            reason="buy executed",
        )
        with patch.object(alerter, "_send_telegram") as mock_send:
            alerter.send_buy_executed(decision)
            mock_send.assert_called_once()
            msg = mock_send.call_args[0][0]
            assert "BTCGBP" in msg

    def test_telegram_failure_does_not_raise(
        self, config: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mock telegram raises error -> no exception propagated."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
        alerter = Alerter(config)

        # httpx is lazily imported inside _send_telegram, so patch at module level
        with patch("httpx.post", side_effect=Exception("Network error")):
            score = CompositeScore(score=60.0, should_buy=False)
            # Should not raise
            alerter.send_score_alert(score, "BTCGBP")
