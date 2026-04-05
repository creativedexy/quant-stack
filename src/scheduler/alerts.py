"""Alert service -- notifications for important pipeline and risk events.

Checks configurable conditions and sends alerts via logging and,
optionally, email/webhook delivery through
:class:`~src.services.alert_delivery.AlertDeliveryService`.

Usage:
    from src.scheduler.alerts import AlertService
    alerts = AlertService(config)
    alerts.check_and_alert(pipeline_result, risk_metrics)
    alerts.check_and_deliver("drawdown_breach", {"max_drawdown": 0.18})
"""

from __future__ import annotations

from typing import Any

from src.services.alert_delivery import AlertDeliveryService
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AlertService:
    """Sends alerts when important events occur.

    Condition checks log at WARNING level.  If delivery channels are
    configured (``alerts.channels`` in settings), the service also
    dispatches via email/webhook through
    :class:`AlertDeliveryService`.
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

        # Delivery service for email / webhook channels
        self._delivery = AlertDeliveryService(alerts_cfg)

    # Persona activation alert names (checked by scheduler)
    PERSONA_ACTIVATION_PENDING = "persona_activation_pending"
    PERSONA_ACTIVATED = "persona_activated"
    PERSONA_AUTO_DEACTIVATED = "persona_auto_deactivated"

    # ------------------------------------------------------------------
    # Existing condition-check interface (unchanged contract)
    # ------------------------------------------------------------------

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

        # -- Pipeline failure ----------------------------------------
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

        # -- Max drawdown exceeded -----------------------------------
        current_drawdown = risk_metrics.get("max_drawdown", 0)
        if current_drawdown > self._max_drawdown:
            msg = (
                f"RISK ALERT: Max drawdown {current_drawdown:.2%} "
                f"exceeds threshold {self._max_drawdown:.2%}"
            )
            triggered.append(msg)

        # -- High correlation ----------------------------------------
        max_corr = risk_metrics.get("max_pairwise_correlation", 0)
        if max_corr > self._max_correlation:
            msg = (
                f"RISK ALERT: Pairwise correlation {max_corr:.2f} "
                f"exceeds threshold {self._max_correlation:.2f}"
            )
            triggered.append(msg)

        # -- Model performance degradation ---------------------------
        current_ic = risk_metrics.get("information_coefficient")
        if current_ic is not None and current_ic < self._min_ic:
            msg = (
                f"MODEL ALERT: Information coefficient {current_ic:.4f} "
                f"below threshold {self._min_ic:.4f}"
            )
            triggered.append(msg)

        # -- Rebalance needed but not executed -----------------------
        if risk_metrics.get("rebalance_needed_not_executed"):
            msg = "REBALANCE ALERT: Rebalance is needed but was not executed"
            triggered.append(msg)

        # -- Send alerts ---------------------------------------------
        for alert_msg in triggered:
            self._send_alert(alert_msg)

        return triggered

    # ------------------------------------------------------------------
    # New delivery-aware interface
    # ------------------------------------------------------------------

    def check_and_deliver(
        self,
        alert_name: str,
        context: dict[str, Any],
    ) -> bool:
        """Check a named condition and deliver via configured channels.

        This is the preferred entry-point for new alert integrations.
        The alert is always logged; delivery (email/webhook) happens
        only when ``alerts.channels`` is configured.

        Args:
            alert_name: Short identifier (e.g. ``"drawdown_breach"``).
            context:    Arbitrary key/value pairs describing the
                        alert (included in webhook payload and email
                        body).

        Returns:
            ``True`` if the alert was triggered AND delivered (or
            if no delivery channels are configured).  ``False`` if
            triggered but delivery failed.
        """
        severity = context.get("severity", "warning")
        message = context.get("message", f"Alert triggered: {alert_name}")

        # Always log
        logger.warning("ALERT [%s]: %s", alert_name, message)

        # Deliver via configured channels
        return self._delivery.deliver(
            alert_name=alert_name,
            severity=severity,
            message=message,
            context=context,
        )

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _send_alert(self, message: str) -> None:
        """Dispatch an alert via all configured methods.

        Args:
            message: The alert message to send.
        """
        if "log" in self._methods:
            logger.warning(f"ALERT: {message}")

        if "email" in self._methods:
            self._delivery.send_email(
                subject="Pipeline Alert",
                body=message,
                recipients=[
                    self._delivery._smtp.get("from_address", "")
                ],
            )

        if "webhook" in self._methods:
            url = self._delivery._webhook_url
            if url:
                self._delivery.send_webhook(
                    {"alert_name": "pipeline", "severity": "warning",
                     "message": message},
                    url,
                )
