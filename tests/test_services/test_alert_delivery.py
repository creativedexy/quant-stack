"""Unit tests for AlertDeliveryService and AlertService delivery wiring.

All external dependencies (smtplib, requests) are mocked -- no
network calls happen in this file.
"""

from __future__ import annotations

import json
import time
from email.mime.multipart import MIMEMultipart
from unittest.mock import MagicMock, call, patch

import pytest

from src.services.alert_delivery import AlertDeliveryService
from src.scheduler.alerts import AlertService


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def smtp_config() -> dict:
    """Minimal alerts config with SMTP settings."""
    return {
        "channels": ["email"],
        "smtp": {
            "host": "smtp.test.local",
            "port": 587,
            "username": "user@test.local",
            "password": "${SMTP_PASSWORD}",
            "from_address": "alerts@test.local",
            "use_tls": True,
        },
        "webhook_url": "",
        "webhook_timeout_s": 5,
    }


@pytest.fixture
def webhook_config() -> dict:
    """Minimal alerts config with webhook settings."""
    return {
        "channels": ["webhook"],
        "smtp": {},
        "webhook_url": "https://hooks.test.local/alert",
        "webhook_timeout_s": 5,
    }


@pytest.fixture
def no_channel_config() -> dict:
    """Alerts config with no delivery channels."""
    return {
        "channels": [],
        "smtp": {},
        "webhook_url": "",
        "webhook_timeout_s": 10,
    }


@pytest.fixture
def full_config(smtp_config: dict) -> dict:
    """Full project config wrapping the alerts section."""
    return {
        "alerts": smtp_config,
        "risk": {"max_drawdown": 0.15, "max_correlation": 0.85},
        "models": {"performance_threshold": {"min_ic": 0.02}},
    }


# ─────────────────────────────────────────────
# Email tests
# ─────────────────────────────────────────────


class TestSendEmail:
    """Tests for AlertDeliveryService.send_email()."""

    @patch.dict("os.environ", {"SMTP_PASSWORD": "s3cret"})
    @patch("src.services.alert_delivery.smtplib.SMTP")
    def test_returns_true_on_success(
        self, mock_smtp_cls: MagicMock, smtp_config: dict
    ) -> None:
        """send_email returns True when SMTP succeeds."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(
            return_value=mock_server
        )
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        svc = AlertDeliveryService(smtp_config)
        result = svc.send_email(
            "Test Alert", "Body text", ["ops@test.local"]
        )

        assert result is True
        mock_server.sendmail.assert_called_once()

    @patch.dict("os.environ", {"SMTP_PASSWORD": "s3cret"})
    @patch("src.services.alert_delivery.smtplib.SMTP")
    def test_returns_false_and_logs_on_smtp_error(
        self, mock_smtp_cls: MagicMock, smtp_config: dict
    ) -> None:
        """send_email returns False and logs ERROR when SMTP raises."""
        mock_smtp_cls.side_effect = ConnectionRefusedError("Connection refused")

        svc = AlertDeliveryService(smtp_config)
        result = svc.send_email(
            "Test Alert", "Body text", ["ops@test.local"]
        )

        assert result is False

    @patch("src.services.alert_delivery.smtplib.SMTP")
    def test_returns_false_when_env_var_missing(
        self, mock_smtp_cls: MagicMock, smtp_config: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """send_email returns False when SMTP_PASSWORD env var is not set."""
        monkeypatch.delenv("SMTP_PASSWORD", raising=False)

        svc = AlertDeliveryService(smtp_config)
        result = svc.send_email(
            "Test Alert", "Body text", ["ops@test.local"]
        )

        assert result is False
        # SMTP should never be called
        mock_smtp_cls.assert_not_called()

    @patch.dict("os.environ", {"SMTP_PASSWORD": "s3cret"})
    @patch("src.services.alert_delivery.smtplib.SMTP")
    def test_subject_prefixed(
        self, mock_smtp_cls: MagicMock, smtp_config: dict
    ) -> None:
        """Email subject is prefixed with [Quant Stack Alert]."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(
            return_value=mock_server
        )
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        svc = AlertDeliveryService(smtp_config)
        svc.send_email("Drawdown", "Body", ["ops@test.local"])

        # Extract the raw email string passed to sendmail
        call_args = mock_server.sendmail.call_args
        raw_msg = call_args[0][2]  # third positional arg
        assert "[Quant Stack Alert] Drawdown" in raw_msg


# ─────────────────────────────────────────────
# Webhook tests
# ─────────────────────────────────────────────


class TestSendWebhook:
    """Tests for AlertDeliveryService.send_webhook()."""

    @patch("src.services.alert_delivery.requests.post")
    def test_returns_true_on_200(
        self, mock_post: MagicMock, webhook_config: dict
    ) -> None:
        """send_webhook returns True when server responds 200."""
        mock_post.return_value = MagicMock(status_code=200)

        svc = AlertDeliveryService(webhook_config)
        result = svc.send_webhook(
            {"alert_name": "test", "severity": "info", "message": "hi"},
            "https://hooks.test.local/alert",
        )

        assert result is True
        mock_post.assert_called_once()

    @patch("src.services.alert_delivery.requests.post")
    def test_returns_false_on_500(
        self, mock_post: MagicMock, webhook_config: dict
    ) -> None:
        """send_webhook returns False when server responds 500."""
        mock_post.return_value = MagicMock(status_code=500)

        svc = AlertDeliveryService(webhook_config)
        result = svc.send_webhook(
            {"alert_name": "test", "severity": "info", "message": "hi"},
            "https://hooks.test.local/alert",
        )

        assert result is False

    @patch("src.services.alert_delivery.requests.post")
    def test_returns_false_on_connection_error(
        self, mock_post: MagicMock, webhook_config: dict
    ) -> None:
        """send_webhook returns False when requests raises ConnectionError."""
        mock_post.side_effect = ConnectionError("Connection refused")

        svc = AlertDeliveryService(webhook_config)
        result = svc.send_webhook(
            {"alert_name": "test", "severity": "info", "message": "hi"},
            "https://hooks.test.local/alert",
        )

        assert result is False

    @patch("src.services.alert_delivery.requests.post")
    def test_payload_contains_required_keys(
        self, mock_post: MagicMock, webhook_config: dict
    ) -> None:
        """Webhook payload always contains source, timestamp, alert_name."""
        mock_post.return_value = MagicMock(status_code=200)

        svc = AlertDeliveryService(webhook_config)
        svc.send_webhook(
            {"alert_name": "drawdown", "severity": "critical",
             "message": "exceeded"},
            "https://hooks.test.local/alert",
        )

        call_kwargs = mock_post.call_args
        sent_payload = call_kwargs.kwargs.get(
            "json", call_kwargs[1].get("json", {})
        )

        assert "source" in sent_payload
        assert sent_payload["source"] == "quant-stack"
        assert "timestamp" in sent_payload
        assert "alert_name" in sent_payload
        assert sent_payload["alert_name"] == "drawdown"

    @patch("src.services.alert_delivery.requests.post")
    def test_timeout_from_config(
        self, mock_post: MagicMock, webhook_config: dict
    ) -> None:
        """Webhook uses timeout from config."""
        mock_post.return_value = MagicMock(status_code=200)

        svc = AlertDeliveryService(webhook_config)
        svc.send_webhook(
            {"alert_name": "test", "severity": "info", "message": "hi"},
            "https://hooks.test.local/alert",
        )

        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs.get(
            "timeout", call_kwargs[1].get("timeout")
        ) == 5


# ─────────────────────────────────────────────
# Deliver (channel routing)
# ─────────────────────────────────────────────


class TestDeliver:
    """Tests for AlertDeliveryService.deliver() channel routing."""

    def test_no_channels_returns_true(
        self, no_channel_config: dict
    ) -> None:
        """No channels configured -> deliver returns True (no-op)."""
        svc = AlertDeliveryService(no_channel_config)
        result = svc.deliver("test_alert", "info", "message")

        assert result is True


# ─────────────────────────────────────────────
# AlertService.check_and_deliver
# ─────────────────────────────────────────────


class TestCheckAndDeliver:
    """Tests for AlertService.check_and_deliver() integration."""

    def test_no_channels_returns_true_without_calling_send(
        self, full_config: dict
    ) -> None:
        """When no channels configured, check_and_deliver returns True
        without calling send_email or send_webhook."""
        # Override channels to empty
        full_config["alerts"]["channels"] = []
        alert_svc = AlertService(full_config)

        with (
            patch.object(alert_svc._delivery, "send_email") as mock_email,
            patch.object(alert_svc._delivery, "send_webhook") as mock_wh,
        ):
            result = alert_svc.check_and_deliver(
                "test_alert",
                {"severity": "info", "message": "just testing"},
            )

        assert result is True
        mock_email.assert_not_called()
        mock_wh.assert_not_called()

    @patch.dict("os.environ", {"SMTP_PASSWORD": "s3cret"})
    @patch("src.services.alert_delivery.smtplib.SMTP")
    def test_email_channel_calls_send_email(
        self, mock_smtp_cls: MagicMock, full_config: dict
    ) -> None:
        """When email channel configured, check_and_deliver invokes
        send_email."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(
            return_value=mock_server
        )
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        full_config["alerts"]["channels"] = ["email"]
        full_config["alerts"]["smtp"]["recipients"] = ["ops@test.local"]

        alert_svc = AlertService(full_config)
        result = alert_svc.check_and_deliver(
            "drawdown_breach",
            {"severity": "critical", "message": "Drawdown 18%"},
        )

        assert result is True
        mock_server.sendmail.assert_called_once()


# ─────────────────────────────────────────────
# Retry behaviour
# ─────────────────────────────────────────────


class TestRetryBehaviour:
    """Tests for retry logic in send_email and send_webhook."""

    @pytest.fixture
    def retry_smtp_config(self) -> dict:
        """Config with retries and zero delay for fast tests."""
        return {
            "channels": ["email"],
            "max_retries": 3,
            "retry_delay_s": 0,
            "min_alert_interval_s": 0,
            "smtp": {
                "host": "smtp.test.local",
                "port": 587,
                "username": "user@test.local",
                "password": "${SMTP_PASSWORD}",
                "from_address": "alerts@test.local",
                "use_tls": True,
            },
            "webhook_url": "",
            "webhook_timeout_s": 5,
        }

    @patch.dict("os.environ", {"SMTP_PASSWORD": "s3cret"})
    @patch("src.services.alert_delivery.smtplib.SMTP")
    def test_email_retries_then_succeeds(
        self, mock_smtp_cls: MagicMock, retry_smtp_config: dict,
    ) -> None:
        """send_email retries on failure and returns True when it works."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(
            return_value=mock_server
        )
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        # Fail twice, then succeed
        mock_server.sendmail.side_effect = [
            ConnectionRefusedError("fail 1"),
            ConnectionRefusedError("fail 2"),
            None,
        ]

        svc = AlertDeliveryService(retry_smtp_config)
        result = svc.send_email("Test", "Body", ["ops@test.local"])

        assert result is True
        assert mock_server.sendmail.call_count == 3

    @patch.dict("os.environ", {"SMTP_PASSWORD": "s3cret"})
    @patch("src.services.alert_delivery.smtplib.SMTP")
    def test_email_retries_exhausted_returns_false(
        self, mock_smtp_cls: MagicMock, retry_smtp_config: dict,
    ) -> None:
        """send_email returns False when all retries are exhausted."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(
            return_value=mock_server
        )
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        mock_server.sendmail.side_effect = ConnectionRefusedError("fail")

        svc = AlertDeliveryService(retry_smtp_config)
        result = svc.send_email("Test", "Body", ["ops@test.local"])

        assert result is False
        assert mock_server.sendmail.call_count == 3

    @patch("src.services.alert_delivery.requests.post")
    def test_webhook_retries_then_succeeds(
        self, mock_post: MagicMock,
    ) -> None:
        """send_webhook retries on failure and returns True when it works."""
        config = {
            "channels": ["webhook"],
            "max_retries": 3,
            "retry_delay_s": 0,
            "min_alert_interval_s": 0,
            "smtp": {},
            "webhook_url": "https://hooks.test.local/alert",
            "webhook_timeout_s": 5,
        }

        # Fail twice, then succeed
        mock_post.side_effect = [
            ConnectionError("fail 1"),
            ConnectionError("fail 2"),
            MagicMock(status_code=200),
        ]

        svc = AlertDeliveryService(config)
        result = svc.send_webhook(
            {"alert_name": "test"}, "https://hooks.test.local/alert",
        )

        assert result is True
        assert mock_post.call_count == 3

    @patch("src.services.alert_delivery.requests.post")
    def test_webhook_retries_exhausted_returns_false(
        self, mock_post: MagicMock,
    ) -> None:
        """send_webhook returns False when all retries are exhausted."""
        config = {
            "channels": ["webhook"],
            "max_retries": 3,
            "retry_delay_s": 0,
            "min_alert_interval_s": 0,
            "smtp": {},
            "webhook_url": "https://hooks.test.local/alert",
            "webhook_timeout_s": 5,
        }

        mock_post.side_effect = ConnectionError("fail")

        svc = AlertDeliveryService(config)
        result = svc.send_webhook(
            {"alert_name": "test"}, "https://hooks.test.local/alert",
        )

        assert result is False
        assert mock_post.call_count == 3


# ─────────────────────────────────────────────
# Rate limiting
# ─────────────────────────────────────────────


class TestRateLimiting:
    """Tests for rate limiting in deliver()."""

    @patch.dict("os.environ", {"SMTP_PASSWORD": "s3cret"})
    @patch("src.services.alert_delivery.smtplib.SMTP")
    def test_second_call_within_interval_is_skipped(
        self, mock_smtp_cls: MagicMock,
    ) -> None:
        """Rapid deliver() calls -- second is rate-limited."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(
            return_value=mock_server
        )
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        config = {
            "channels": ["email"],
            "max_retries": 1,
            "retry_delay_s": 0,
            "min_alert_interval_s": 60,
            "smtp": {
                "host": "smtp.test.local",
                "port": 587,
                "username": "user@test.local",
                "password": "${SMTP_PASSWORD}",
                "from_address": "alerts@test.local",
                "use_tls": True,
                "recipients": ["ops@test.local"],
            },
            "webhook_url": "",
            "webhook_timeout_s": 5,
        }

        svc = AlertDeliveryService(config)

        # First call goes through
        svc.deliver("alert_a", "info", "first message")
        first_count = mock_server.sendmail.call_count

        # Second call within interval -- rate-limited
        svc.deliver("alert_b", "info", "second message")
        second_count = mock_server.sendmail.call_count

        assert first_count == 1
        assert second_count == 1  # No additional call


# ─────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────


class TestDeduplication:
    """Tests for alert deduplication in deliver()."""

    def test_duplicate_subject_is_suppressed(self) -> None:
        """Same subject within dedup window returns True without sending."""
        config = {
            "channels": ["webhook"],
            "max_retries": 1,
            "retry_delay_s": 0,
            "min_alert_interval_s": 0,  # No rate limiting
            "smtp": {},
            "webhook_url": "https://hooks.test.local/alert",
            "webhook_timeout_s": 5,
        }
        svc = AlertDeliveryService(config)

        # Set a long dedup window so it catches the duplicate
        svc._min_alert_interval = 60.0

        with patch("src.services.alert_delivery.requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)

            # First deliver
            result1 = svc.deliver("drawdown", "critical", "msg1")
            # Second identical alert
            result2 = svc.deliver("drawdown", "critical", "msg2")

        # Both succeed but only one webhook call
        assert result1 is True
        assert result2 is True
        assert mock_post.call_count == 1


# ─────────────────────────────────────────────
# HTML email
# ─────────────────────────────────────────────


class TestHtmlEmail:
    """Tests for HTML email support."""

    @patch.dict("os.environ", {"SMTP_PASSWORD": "s3cret"})
    @patch("src.services.alert_delivery.smtplib.SMTP")
    def test_html_email_sends_multipart(
        self, mock_smtp_cls: MagicMock,
    ) -> None:
        """When html_emails is True, email is MIMEMultipart."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(
            return_value=mock_server
        )
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        config = {
            "channels": ["email"],
            "max_retries": 1,
            "retry_delay_s": 0,
            "min_alert_interval_s": 0,
            "html_emails": True,
            "smtp": {
                "host": "smtp.test.local",
                "port": 587,
                "username": "user@test.local",
                "password": "${SMTP_PASSWORD}",
                "from_address": "alerts@test.local",
                "use_tls": True,
            },
            "webhook_url": "",
            "webhook_timeout_s": 5,
        }

        svc = AlertDeliveryService(config)
        result = svc.send_email("Test", "Body text", ["ops@test.local"])

        assert result is True
        call_args = mock_server.sendmail.call_args
        raw_msg = call_args[0][2]
        # Multipart messages contain Content-Type: multipart/alternative
        assert "multipart/alternative" in raw_msg
