"""Alert delivery service -- email and webhook dispatch.

Handles sending alert notifications via SMTP email and/or HTTP
webhook.  Both channels are independently testable and never raise
on failure (they return ``False`` and log at ERROR level instead).

Features:
- Retry with exponential backoff on transient failures
- Rate limiting per channel to prevent alert spam
- Deduplication window for repeated alert subjects
- Optional HTML email alongside plain text

Usage:
    from src.services.alert_delivery import AlertDeliveryService
    svc = AlertDeliveryService(config["alerts"])
    svc.send_email("Subject", "Body text", ["ops@example.com"])
    svc.send_webhook({"alert_name": "drawdown", "message": "..."}, url)
"""

from __future__ import annotations

import os
import smtplib
import time
from collections import deque
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Maximum number of recent subjects to track for deduplication
_DEDUP_HISTORY_SIZE = 100


class AlertDeliveryService:
    """Delivers alert notifications via email and/or webhook.

    Both delivery methods return ``True`` on success and ``False``
    on failure.  Failures are logged at ERROR level but never
    raised -- the caller decides whether to retry.

    Args:
        config: The ``alerts`` section of ``config/settings.yaml``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._channels: list[str] = config.get("channels", [])
        self._smtp = config.get("smtp", {})
        self._webhook_url: str = config.get("webhook_url", "")
        self._webhook_timeout: int = config.get("webhook_timeout_s", 10)

        # Retry settings
        self._max_retries: int = config.get("max_retries", 3)
        self._retry_delay: float = config.get("retry_delay_s", 2.0)

        # Rate limiting
        self._min_alert_interval: float = config.get(
            "min_alert_interval_s", 60.0,
        )
        self._last_send_time: dict[str, float] = {}

        # Deduplication
        self._recent_subjects: deque[tuple[str, float]] = deque(
            maxlen=_DEDUP_HISTORY_SIZE,
        )

        # HTML email toggle
        self._html_emails: bool = config.get("html_emails", False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_email(
        self,
        subject: str,
        body: str,
        recipients: list[str],
    ) -> bool:
        """Send an email via SMTP with retry logic.

        The subject is automatically prefixed with
        ``[Quant Stack Alert]``.

        Args:
            subject:    Email subject line.
            body:       Plain-text email body.
            recipients: List of recipient email addresses.

        Returns:
            ``True`` if the email was sent successfully, ``False``
            otherwise.
        """
        prefixed_subject = f"[Quant Stack Alert] {subject}"

        # Resolve SMTP password (may be an env-var reference)
        password = self._resolve_env_var(self._smtp.get("password", ""))
        if password is None:
            return False

        host = self._smtp.get("host", "localhost")
        port = self._smtp.get("port", 587)
        username = self._smtp.get("username", "")
        from_addr = self._smtp.get("from_address", username)
        use_tls = self._smtp.get("use_tls", True)

        for attempt in range(1, self._max_retries + 1):
            try:
                if self._html_emails:
                    msg = self._build_html_message(
                        prefixed_subject, body, from_addr, recipients,
                    )
                else:
                    msg = MIMEText(body, "plain")
                    msg["Subject"] = prefixed_subject
                    msg["From"] = from_addr
                    msg["To"] = ", ".join(recipients)

                with smtplib.SMTP(host, port) as server:
                    if use_tls:
                        server.starttls()
                    if username and password:
                        server.login(username, password)
                    server.sendmail(from_addr, recipients, msg.as_string())

                logger.info(
                    "Email sent: %s -> %s", prefixed_subject, recipients,
                )
                return True

            except Exception as exc:
                logger.warning(
                    "Email attempt %d/%d failed: %s",
                    attempt, self._max_retries, exc,
                )
                if attempt < self._max_retries:
                    backoff = self._retry_delay * (2 ** (attempt - 1))
                    time.sleep(backoff)

        logger.error(
            "Email delivery failed after %d attempts", self._max_retries,
        )
        return False

    def send_webhook(
        self,
        payload: dict[str, Any],
        url: str,
    ) -> bool:
        """Send an alert payload via HTTP POST with retry logic.

        The payload is augmented with ``source`` and ``timestamp``
        keys before dispatch.

        Args:
            payload: Alert data dict.  Must contain at least
                     ``alert_name``.
            url:     Webhook endpoint URL.

        Returns:
            ``True`` if the remote server returned HTTP 200,
            ``False`` otherwise.
        """
        enriched: dict[str, Any] = {
            "source": "quant-stack",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    json=enriched,
                    timeout=self._webhook_timeout,
                )
                if resp.status_code == 200:
                    logger.info("Webhook delivered to %s", url)
                    return True

                logger.warning(
                    "Webhook attempt %d/%d returned status %d from %s",
                    attempt, self._max_retries, resp.status_code, url,
                )

            except Exception as exc:
                logger.warning(
                    "Webhook attempt %d/%d failed: %s",
                    attempt, self._max_retries, exc,
                )

            if attempt < self._max_retries:
                backoff = self._retry_delay * (2 ** (attempt - 1))
                time.sleep(backoff)

        logger.error(
            "Webhook delivery failed after %d attempts", self._max_retries,
        )
        return False

    def deliver(
        self,
        alert_name: str,
        severity: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Deliver an alert via all configured channels.

        Applies rate limiting and deduplication before sending.
        If no channels are configured the alert is silently
        suppressed (returns ``True`` -- not an error condition).

        Args:
            alert_name: Short identifier for the alert.
            severity:   e.g. "warning", "critical".
            message:    Human-readable description.
            context:    Extra key/value pairs merged into the
                        webhook payload.

        Returns:
            ``True`` if all configured channels succeeded (or none
            are configured), ``False`` if any channel failed.
        """
        if not self._channels:
            logger.info(
                "No delivery channels configured -- alert suppressed"
            )
            return True

        subject = f"{severity.upper()}: {alert_name}"

        # Deduplication check
        if self._is_duplicate(subject):
            logger.info(
                "Alert suppressed (duplicate): %s", subject,
            )
            return True

        ctx = context or {}
        all_ok = True

        if "email" in self._channels:
            if self._is_rate_limited("email"):
                logger.info("Email rate-limited -- skipping")
            else:
                recipients = self._smtp.get("recipients", [])
                from_addr = self._smtp.get("from_address", "")
                if not recipients and from_addr:
                    recipients = [from_addr]
                ok = self.send_email(
                    subject=subject,
                    body=message,
                    recipients=recipients,
                )
                if ok:
                    self._record_send_time("email")
                else:
                    all_ok = False

        if "webhook" in self._channels:
            if self._is_rate_limited("webhook"):
                logger.info("Webhook rate-limited -- skipping")
            else:
                url = self._webhook_url
                if url:
                    payload: dict[str, Any] = {
                        "alert_name": alert_name,
                        "severity": severity,
                        "message": message,
                        **ctx,
                    }
                    ok = self.send_webhook(payload, url)
                    if ok:
                        self._record_send_time("webhook")
                    else:
                        all_ok = False
                else:
                    logger.error(
                        "Webhook channel enabled but no URL configured"
                    )
                    all_ok = False

        self._record_subject(subject)
        return all_ok

    # ------------------------------------------------------------------
    # Rate limiting and deduplication
    # ------------------------------------------------------------------

    def _is_rate_limited(self, channel: str) -> bool:
        """Return True if *channel* was used too recently."""
        last = self._last_send_time.get(channel, 0.0)
        return (time.monotonic() - last) < self._min_alert_interval

    def _record_send_time(self, channel: str) -> None:
        """Record the current time as the last send for *channel*."""
        self._last_send_time[channel] = time.monotonic()

    def _is_duplicate(self, subject: str) -> bool:
        """Return True if *subject* was sent within the dedup window."""
        now = time.monotonic()
        for subj, ts in self._recent_subjects:
            if subj == subject and (now - ts) < self._min_alert_interval:
                return True
        return False

    def _record_subject(self, subject: str) -> None:
        """Record a subject and timestamp for deduplication."""
        self._recent_subjects.append((subject, time.monotonic()))

    # ------------------------------------------------------------------
    # HTML email builder
    # ------------------------------------------------------------------

    def _build_html_message(
        self,
        subject: str,
        body: str,
        from_addr: str,
        recipients: list[str],
    ) -> MIMEMultipart:
        """Build a multipart email with plain text and HTML parts."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(recipients)

        # Plain text part
        msg.attach(MIMEText(body, "plain"))

        # HTML part
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        html = (
            "<html><body>"
            '<div style="font-family:sans-serif;max-width:600px;'
            'margin:0 auto;padding:20px">'
            f'<h2 style="color:#1e293b;border-bottom:2px solid #3b82f6;'
            f'padding-bottom:8px">{subject}</h2>'
            f'<p style="color:#334155;line-height:1.6;white-space:pre-wrap">'
            f"{body}</p>"
            f'<p style="color:#94a3b8;font-size:12px;margin-top:24px;'
            f'border-top:1px solid #e2e8f0;padding-top:8px">'
            f"Quant Stack Alert | {timestamp}</p>"
            "</div></body></html>"
        )
        msg.attach(MIMEText(html, "html"))

        return msg

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_env_var(self, value: str) -> str | None:
        """Resolve a ``${VAR}`` reference to an environment variable.

        Returns the resolved string, or ``None`` if the env var is
        missing (logs ERROR in that case).  If *value* does not
        start with ``${``, it is returned as-is.
        """
        if not isinstance(value, str):
            return value

        if value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            resolved = os.environ.get(var_name)
            if resolved is None:
                logger.error(
                    "Environment variable %s not set -- cannot send email",
                    var_name,
                )
                return None
            return resolved

        return value
