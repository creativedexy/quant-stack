"""Tests for src/services/deployment_service.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.deployment import SizedPosition, TradeOrder
from src.services.deployment_service import DeploymentService


@pytest.fixture
def service(deployment_config) -> DeploymentService:
    """DeploymentService with test config."""
    return DeploymentService(config=deployment_config)


@pytest.fixture
def service_with_orders(deployment_config) -> DeploymentService:
    """DeploymentService with pre-generated orders."""
    svc = DeploymentService(config=deployment_config)
    positions = [
        SizedPosition(
            ticker="NVDA", name="NVIDIA", exchange="NASDAQ",
            sector="Technology", gbp_price=100.0, shares=5,
            gbp_amount=500.0, pct_of_capital=0.10,
            order_type="LIMIT", composite_score=0.75,
        ),
        SizedPosition(
            ticker="XOM", name="Exxon", exchange="NYSE",
            sector="Energy", gbp_price=85.0, shares=3,
            gbp_amount=255.0, pct_of_capital=0.05,
            order_type="LIMIT", composite_score=0.65,
        ),
    ]
    queue = svc._ensure_queue()
    queue.generate_orders(positions)
    return svc


# ------------------------------------------------------------------
# Service initialisation
# ------------------------------------------------------------------


class TestServiceInit:
    """Tests for service initialisation."""

    def test_default_state_is_idle(self, service):
        """Service starts in IDLE state."""
        status = service.get_cycle_status()
        assert status["state"] == "IDLE"

    def test_scored_candidates_initially_empty(self, service):
        """No scored candidates before pipeline runs."""
        assert service.get_scored_candidates() == []

    def test_lazy_config_loading(self):
        """Config is loaded lazily if not provided."""
        svc = DeploymentService(config=None)
        # Accessing config triggers load -- may fail if file missing,
        # but should not fail at init.
        assert svc._config is None


# ------------------------------------------------------------------
# Scored candidates
# ------------------------------------------------------------------


class TestScoredCandidates:
    """Tests for scored candidate caching."""

    def test_set_and_get_scored(self, service):
        """Scored candidates round-trip through set/get."""
        data = [{"ticker": "NVDA", "score": 0.8}]
        service.set_scored_candidates(data)
        assert service.get_scored_candidates() == data

    def test_set_scored_converts_dataclasses(self, service):
        """Dataclass objects are converted to dicts."""
        from src.deployment import ScoredCandidate
        sc = ScoredCandidate(
            ticker="TEST", name="Test", exchange="NYSE",
            sector="Tech", market_cap_usd=10e9,
            source_universe="us_growth",
            scores={"composite": 0.7},
        )
        service.set_scored_candidates([sc])
        result = service.get_scored_candidates()
        assert len(result) == 1
        assert result[0]["ticker"] == "TEST"


# ------------------------------------------------------------------
# Trade queue operations
# ------------------------------------------------------------------


class TestTradeQueueOps:
    """Tests for trade queue operations via service."""

    def test_get_trade_queue(self, service_with_orders):
        """Trade queue returns generated orders."""
        queue = service_with_orders.get_trade_queue()
        assert len(queue) == 2

    def test_approve_order(self, service_with_orders):
        """Approving an order returns True."""
        queue = service_with_orders.get_trade_queue()
        order_id = queue[0]["order_id"]
        assert service_with_orders.approve_order(order_id) is True

    def test_reject_order(self, service_with_orders):
        """Rejecting an order returns True."""
        queue = service_with_orders.get_trade_queue()
        order_id = queue[0]["order_id"]
        assert service_with_orders.reject_order(order_id, "No") is True

    def test_approve_all(self, service_with_orders):
        """Bulk approve returns count."""
        count = service_with_orders.approve_all_orders()
        assert count == 2

    def test_confirm_and_submit_dry_run(self, service_with_orders):
        """Dry-run submission returns results."""
        service_with_orders.approve_all_orders()
        results = service_with_orders.confirm_and_submit(dry_run=True)
        assert len(results) == 2
        for r in results:
            assert r["status"] == "SUBMITTED"

    def test_queue_summary(self, service_with_orders):
        """Queue summary has expected structure."""
        summary = service_with_orders.get_queue_summary()
        assert "total_orders" in summary
        assert summary["total_orders"] == 2


# ------------------------------------------------------------------
# Emergency stop
# ------------------------------------------------------------------


class TestEmergencyStop:
    """Tests for emergency stop via service."""

    def test_emergency_stop_halts(self, service_with_orders):
        """Emergency stop sets state to HALTED."""
        result = service_with_orders.emergency_stop()
        assert result["state"] == "HALTED"
        assert result["orders_cancelled"] == 2

    def test_cycle_status_after_halt(self, service_with_orders):
        """Cycle status reflects HALTED after emergency stop."""
        service_with_orders.emergency_stop()
        status = service_with_orders.get_cycle_status()
        assert status["state"] == "HALTED"


# ------------------------------------------------------------------
# IB integration
# ------------------------------------------------------------------


class TestIBIntegration:
    """Tests for IB buying power queries."""

    def test_no_broker_returns_none(self, service):
        """Without broker, buying power returns None."""
        assert service.get_ib_buying_power() is None

    def test_broker_returns_value(self, deployment_config):
        """With broker mock, buying power returns value."""
        mock_broker = MagicMock()
        mock_broker.get_account_value.return_value = 5000.0
        svc = DeploymentService(
            config=deployment_config, broker=mock_broker,
        )
        assert svc.get_ib_buying_power() == 5000.0

    def test_broker_error_returns_none(self, deployment_config):
        """Broker error returns None gracefully."""
        mock_broker = MagicMock()
        mock_broker.get_account_value.side_effect = RuntimeError("fail")
        svc = DeploymentService(
            config=deployment_config, broker=mock_broker,
        )
        assert svc.get_ib_buying_power() is None
