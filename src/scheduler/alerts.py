"""Alert service — notifications for important pipeline and risk events.

Checks configurable conditions and sends alerts via logging (with stubs
for email and webhook delivery in future).

Usage:
    from src.scheduler.alerts import AlertService
    alerts = AlertService(config)
    alerts.check_and_alert(pipeline_result, risk_metrics)
"""

from __future__ import annotations

from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class AlertService:
    """Sends alerts when important events occur.

    Currently logs all alerts at WARNING level.  Email and webhook
    delivery methods are stubbed for future implementation.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialise the alert service.

        Args:
            config: Project configuration dict. If None, loads from
                    config/settings.yaml.
        """
        if config is None:
            from src.utils.config import load_config
            config = load_config()

        self.config = config
        risk_cfg = config.get("risk", {})
        self._max_drawdown = risk_cfg.get("max_drawdown", 0.15)
        self._max_correlation = risk_cfg.get("max_correlation", 0.85)

        models_cfg = config.get("models", {})
        perf_cfg = models_cfg.get("performance_threshold", {})
        self._min_ic = perf_cfg.get("min_ic", 0.02)

        alerts_cfg = config.get("alerts", {})
        self._enabled = alerts_cfg.get("enabled", True)
        self._methods = alerts_cfg.get("methods", ["log"])

    def check_and_alert(
        self,
        pipeline_result: dict[str, Any],
        risk_metrics: dict[str, Any],
    ) -> list[str]:
        """Check alert conditions and send notifications.

        Conditions checked:
        - Pipeline failure (any ticker failed to update)
        - Max drawdown exceeded
        - High correlation detected between holdings
        - Model performance degradation (IC below threshold)
        - Rebalance needed but not executed

        Args:
            pipeline_result: Output from PipelineRunner.run_daily().
            risk_metrics: Dictionary of current risk metrics.

        Returns:
            List of alert messages that were triggered.
        """
        if not self._enabled:
            return []

        triggered: list[str] = []

        # ── Pipeline failure ───────────────────────────────────────
        if pipeline_result.get("status") == "failed":
            msg = (
                "PIPELINE FAILURE: All tickers failed to update. "
                f"Errors: {pipeline_result.get('errors', [])}"
            )
            triggered.append(msg)

        elif pipeline_result.get("tickers_failed"):
            failed = pipeline_result["tickers_failed"]
            msg = (
                f"PIPELINE PARTIAL: {len(failed)} ticker(s) failed to update: "
                f"{failed}"
            )
            triggered.append(msg)

        # ── Max drawdown exceeded ──────────────────────────────────
        current_drawdown = risk_metrics.get("max_drawdown", 0)
        if current_drawdown > self._max_drawdown:
            msg = (
                f"RISK ALERT: Max drawdown {current_drawdown:.2%} "
                f"exceeds threshold {self._max_drawdown:.2%}"
            )
            triggered.append(msg)

        # ── High correlation ───────────────────────────────────────
        max_corr = risk_metrics.get("max_pairwise_correlation", 0)
        if max_corr > self._max_correlation:
            msg = (
                f"RISK ALERT: Pairwise correlation {max_corr:.2f} "
                f"exceeds threshold {self._max_correlation:.2f}"
            )
            triggered.append(msg)

        # ── Model performance degradation ──────────────────────────
        current_ic = risk_metrics.get("information_coefficient")
        if current_ic is not None and current_ic < self._min_ic:
            msg = (
                f"MODEL ALERT: Information coefficient {current_ic:.4f} "
                f"below threshold {self._min_ic:.4f}"
            )
            triggered.append(msg)

        # ── Rebalance needed but not executed ──────────────────────
        if risk_metrics.get("rebalance_needed_not_executed"):
            msg = "REBALANCE ALERT: Rebalance is needed but was not executed"
            triggered.append(msg)

        # ── Send alerts ────────────────────────────────────────────
        for alert_msg in triggered:
            self._send_alert(alert_msg)

        return triggered

    def _send_alert(self, message: str) -> None:
        """Dispatch an alert via all configured methods.

        Args:
            message: The alert message to send.
        """
        if "log" in self._methods:
            logger.warning(f"ALERT: {message}")

        if "email" in self._methods:
            self._email_alert(message)

        if "webhook" in self._methods:
            self._webhook_alert(message)

    def _email_alert(self, message: str) -> None:
        """Send an alert via email (stub for future implementation).

        Args:
            message: The alert message to send.
        """
        # TODO: Implement SMTP email delivery using config['alerts']['email']
        logger.debug(f"Email alert stub: {message}")

    def _webhook_alert(self, message: str) -> None:
        """Send an alert via webhook (stub for future implementation).

        Args:
            message: The alert message to send.
        """
        # TODO: Implement webhook POST using config['alerts']['webhook']['url']
        logger.debug(f"Webhook alert stub: {message}")
