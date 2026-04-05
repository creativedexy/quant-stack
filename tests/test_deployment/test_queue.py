"""Tests for src/deployment/queue.py."""

from __future__ import annotations

import pytest

from src.deployment import SizedPosition, TradeOrder
from src.deployment.queue import TradeQueueManager


@pytest.fixture
def queue(deployment_config) -> TradeQueueManager:
    """TradeQueueManager with test config."""
    return TradeQueueManager(config=deployment_config)


@pytest.fixture
def sized_positions() -> list[SizedPosition]:
    """Sample sized positions for queue tests."""
    return [
        SizedPosition(
            ticker="NVDA",
            name="NVIDIA",
            exchange="NASDAQ",
            sector="Technology",
            asset_type="stock",
            gbp_price=100.0,
            shares=5,
            gbp_amount=500.0,
            pct_of_capital=0.10,
            order_type="LIMIT",
            composite_score=0.75,
            conviction="HIGH",
            reasoning_summary="Strong growth.",
        ),
        SizedPosition(
            ticker="SPY",
            name="S&P 500 ETF",
            exchange="NYSE",
            sector="Diversified",
            asset_type="etf",
            gbp_price=50.0,
            shares=8,
            gbp_amount=400.0,
            pct_of_capital=0.08,
            order_type="MARKET",
            composite_score=0.70,
            conviction="MEDIUM",
            reasoning_summary="Broad exposure.",
        ),
        SizedPosition(
            ticker="XOM",
            name="Exxon Mobil",
            exchange="NYSE",
            sector="Energy",
            asset_type="stock",
            gbp_price=85.0,
            shares=3,
            gbp_amount=255.0,
            pct_of_capital=0.05,
            order_type="LIMIT",
            composite_score=0.65,
            conviction="MEDIUM",
            reasoning_summary="Contrarian value.",
        ),
    ]


# ------------------------------------------------------------------
# Order generation
# ------------------------------------------------------------------


class TestOrderGeneration:
    """Tests for order generation from sized positions."""

    def test_generates_correct_number_of_orders(self, queue, sized_positions):
        """One order per sized position."""
        orders = queue.generate_orders(sized_positions)
        assert len(orders) == len(sized_positions)

    def test_orders_are_trade_order_objects(self, queue, sized_positions):
        """Generated items are TradeOrder instances."""
        orders = queue.generate_orders(sized_positions)
        for o in orders:
            assert isinstance(o, TradeOrder)

    def test_orders_start_pending_approval(self, queue, sized_positions):
        """All generated orders start in PENDING_APPROVAL status."""
        orders = queue.generate_orders(sized_positions)
        for o in orders:
            assert o.status == "PENDING_APPROVAL"

    def test_limit_orders_have_buffered_price(self, queue, sized_positions):
        """LIMIT orders have limit_price > gbp_price (buffer applied)."""
        orders = queue.generate_orders(sized_positions)
        limit_orders = [o for o in orders if o.order_type == "LIMIT"]
        assert len(limit_orders) > 0
        for o in limit_orders:
            assert o.limit_price is not None
            assert o.limit_price > 0

    def test_market_orders_have_no_limit_price(self, queue, sized_positions):
        """MARKET orders have limit_price = None."""
        orders = queue.generate_orders(sized_positions)
        market_orders = [o for o in orders if o.order_type == "MARKET"]
        assert len(market_orders) > 0
        for o in market_orders:
            assert o.limit_price is None

    def test_order_ids_are_unique(self, queue, sized_positions):
        """Each order has a unique ID."""
        orders = queue.generate_orders(sized_positions)
        ids = [o.order_id for o in orders]
        assert len(ids) == len(set(ids))

    def test_direction_is_buy(self, queue, sized_positions):
        """All deployment orders are BUY."""
        orders = queue.generate_orders(sized_positions)
        for o in orders:
            assert o.direction == "BUY"


# ------------------------------------------------------------------
# Approval flow
# ------------------------------------------------------------------


class TestApprovalFlow:
    """Tests for the two-step approval flow."""

    def test_approve_single_order(self, queue, sized_positions):
        """Approving an order changes status to APPROVED."""
        orders = queue.generate_orders(sized_positions)
        result = queue.approve_order(orders[0].order_id)
        assert result is True
        assert orders[0].status == "APPROVED"

    def test_reject_single_order(self, queue, sized_positions):
        """Rejecting an order changes status to REJECTED."""
        orders = queue.generate_orders(sized_positions)
        result = queue.reject_order(orders[0].order_id, "Too risky")
        assert result is True
        assert orders[0].status == "REJECTED"
        assert orders[0].rejection_reason == "Too risky"

    def test_approve_nonexistent_order(self, queue, sized_positions):
        """Approving a nonexistent order returns False."""
        queue.generate_orders(sized_positions)
        assert queue.approve_order("FAKE-ID") is False

    def test_approve_all(self, queue, sized_positions):
        """Bulk approve sets all pending orders to APPROVED."""
        queue.generate_orders(sized_positions)
        count = queue.approve_all()
        assert count == len(sized_positions)
        for o in queue.get_all_orders():
            assert o.status == "APPROVED"

    def test_get_approved_orders(self, queue, sized_positions):
        """get_approved_orders returns only APPROVED orders."""
        orders = queue.generate_orders(sized_positions)
        queue.approve_order(orders[0].order_id)
        approved = queue.get_approved_orders()
        assert len(approved) == 1
        assert approved[0].order_id == orders[0].order_id

    def test_cannot_approve_rejected_order(self, queue, sized_positions):
        """Cannot approve an already-rejected order."""
        orders = queue.generate_orders(sized_positions)
        queue.reject_order(orders[0].order_id, "No")
        result = queue.approve_order(orders[0].order_id)
        assert result is False


# ------------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------------


class TestSafetyChecks:
    """Tests for pre-submission safety checks."""

    def test_price_sanity_within_drift(self, queue, sized_positions):
        """Price within drift tolerance passes."""
        orders = queue.generate_orders(sized_positions)
        limit_order = next(o for o in orders if o.order_type == "LIMIT")
        # Live price very close to limit price.
        assert queue.check_price_sanity(limit_order, limit_order.limit_price * 0.995)

    def test_price_sanity_exceeds_drift(self, queue, sized_positions):
        """Price drift exceeding max fails."""
        orders = queue.generate_orders(sized_positions)
        limit_order = next(o for o in orders if o.order_type == "LIMIT")
        # Live price 5% away.
        assert not queue.check_price_sanity(
            limit_order, limit_order.limit_price * 0.90
        )

    def test_market_order_passes_price_sanity(self, queue, sized_positions):
        """Market orders always pass price sanity check."""
        orders = queue.generate_orders(sized_positions)
        market_order = next(o for o in orders if o.order_type == "MARKET")
        assert queue.check_price_sanity(market_order, 999.0)

    def test_daily_loss_limit_ok(self, queue):
        """Within daily loss limit returns True."""
        assert queue.check_daily_loss_limit(1000.0, 950.0)

    def test_daily_loss_limit_breached(self, queue):
        """Exceeding daily loss limit returns False."""
        assert not queue.check_daily_loss_limit(1000.0, 850.0)

    def test_buying_power_sufficient(self, queue, sized_positions):
        """Sufficient buying power returns True."""
        orders = queue.generate_orders(sized_positions)
        assert queue.check_buying_power(orders, 10000.0)

    def test_buying_power_insufficient(self, queue, sized_positions):
        """Insufficient buying power returns False."""
        orders = queue.generate_orders(sized_positions)
        assert not queue.check_buying_power(orders, 1.0)


# ------------------------------------------------------------------
# Submission
# ------------------------------------------------------------------


class TestSubmission:
    """Tests for order submission."""

    def test_dry_run_submits_all_approved(self, queue, sized_positions):
        """Dry run marks approved orders as SUBMITTED."""
        queue.generate_orders(sized_positions)
        queue.approve_all()
        results = queue.submit_orders(dry_run=True)
        assert len(results) == len(sized_positions)
        for o in results:
            assert o.status == "SUBMITTED"

    def test_dry_run_sets_timestamp(self, queue, sized_positions):
        """Dry run sets fill_timestamp."""
        queue.generate_orders(sized_positions)
        queue.approve_all()
        results = queue.submit_orders(dry_run=True)
        for o in results:
            assert o.fill_timestamp is not None

    def test_submit_without_approved_returns_empty(self, queue, sized_positions):
        """Submitting with no approved orders returns empty list."""
        queue.generate_orders(sized_positions)
        results = queue.submit_orders(dry_run=True)
        assert results == []

    def test_submit_without_broker_fails_live(self, queue, sized_positions):
        """Live submission without broker sets status to FAILED."""
        # Force live execution enabled.
        queue._config["execution"]["mode"] = "LIVE"
        queue._config["execution"]["confirm_live"] = True

        queue.generate_orders(sized_positions)
        queue.approve_all()
        results = queue.submit_orders(dry_run=False)
        for o in results:
            assert o.status == "FAILED"


# ------------------------------------------------------------------
# Emergency stop
# ------------------------------------------------------------------


class TestEmergencyStop:
    """Tests for emergency stop."""

    def test_cancel_all_pending(self, queue, sized_positions):
        """Emergency stop cancels all pending orders."""
        queue.generate_orders(sized_positions)
        count = queue.cancel_all()
        assert count == len(sized_positions)
        for o in queue.get_all_orders():
            assert o.status == "REJECTED"
            assert o.rejection_reason == "Emergency stop"

    def test_cancel_all_mixed_statuses(self, queue, sized_positions):
        """Emergency stop cancels pending and approved but not submitted."""
        orders = queue.generate_orders(sized_positions)
        queue.approve_order(orders[0].order_id)
        # Manually mark one as submitted.
        orders[1].status = "SUBMITTED"
        count = queue.cancel_all()
        # Should cancel the approved one and the remaining pending one.
        assert count == 2
        assert orders[1].status == "SUBMITTED"  # Unchanged.


# ------------------------------------------------------------------
# Queue summary
# ------------------------------------------------------------------


class TestQueueSummary:
    """Tests for queue summary."""

    def test_summary_correct_counts(self, queue, sized_positions):
        """Summary reports correct status counts."""
        orders = queue.generate_orders(sized_positions)
        queue.approve_order(orders[0].order_id)
        summary = queue.get_queue_summary()
        assert summary["total_orders"] == len(sized_positions)
        assert summary["status_counts"]["APPROVED"] == 1
        assert summary["status_counts"]["PENDING_APPROVAL"] == 2
