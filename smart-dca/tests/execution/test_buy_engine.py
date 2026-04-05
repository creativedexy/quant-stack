"""Tests for BuyEngine."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.execution.buy_engine import BuyEngine
from src.execution.order_manager import OrderResult
from src.signals.scorer import CompositeScore
from src.utils.exceptions import OrderError


@pytest.fixture
def config(sample_settings: dict) -> dict:
    """Add assets to sample settings for BuyEngine."""
    cfg = dict(sample_settings)
    cfg["assets"] = {
        "BTCGBP": {
            "weekly_allocation_gbp": 10.0,
            "min_order_gbp": 5.0,
            "enabled": True,
        },
        "ETHGBP": {
            "weekly_allocation_gbp": 10.0,
            "min_order_gbp": 5.0,
            "enabled": True,
        },
        "XRPGBP": {
            "weekly_allocation_gbp": 5.0,
            "min_order_gbp": 2.0,
            "enabled": True,
        },
    }
    return cfg


@pytest.fixture
def mock_scorer() -> MagicMock:
    scorer = MagicMock()
    scorer.compute.return_value = CompositeScore(score=80.0, should_buy=True)
    return scorer


@pytest.fixture
def mock_budget() -> MagicMock:
    budget = MagicMock()
    budget.get_remaining.return_value = 10.0
    budget.can_spend.return_value = True
    return budget


@pytest.fixture
def mock_order_mgr() -> MagicMock:
    mgr = MagicMock()
    mgr.place_limit_buy.return_value = OrderResult(
        order_id="99999",
        asset="BTCGBP",
        status="NEW",
        fill_price=50000.0,
        quantity=0.0002,
        amount_gbp=10.0,
        timestamp=datetime.now(timezone.utc),
        raw_response={},
    )
    return mgr


@pytest.fixture
def mock_price_feed() -> MagicMock:
    feed = MagicMock()
    feed.get_midprice.return_value = 50000.0
    return feed


@pytest.fixture
def engine(
    mock_scorer: MagicMock,
    mock_budget: MagicMock,
    mock_order_mgr: MagicMock,
    mock_price_feed: MagicMock,
    config: dict,
) -> BuyEngine:
    return BuyEngine(
        scorer=mock_scorer,
        budget_tracker=mock_budget,
        order_manager=mock_order_mgr,
        price_feed=mock_price_feed,
        config=config,
    )


class TestBuyEngine:
    def test_score_below_threshold_no_order(
        self,
        engine: BuyEngine,
        mock_scorer: MagicMock,
        mock_order_mgr: MagicMock,
    ) -> None:
        """score 50, threshold 65 -> executed False, place_limit_buy never called."""
        mock_scorer.compute.return_value = CompositeScore(score=50.0, should_buy=False)
        decision = engine.evaluate_and_buy("BTCGBP")
        assert decision.executed is False
        assert "below threshold" in decision.reason
        mock_order_mgr.place_limit_buy.assert_not_called()

    def test_budget_exhausted_no_order(
        self,
        engine: BuyEngine,
        mock_budget: MagicMock,
        mock_order_mgr: MagicMock,
    ) -> None:
        """score 80, remaining 0.0 -> executed False."""
        mock_budget.get_remaining.return_value = 0.0
        decision = engine.evaluate_and_buy("BTCGBP")
        assert decision.executed is False
        assert "budget exhausted" in decision.reason
        mock_order_mgr.place_limit_buy.assert_not_called()

    def test_successful_buy_records_spend(
        self,
        engine: BuyEngine,
        mock_budget: MagicMock,
    ) -> None:
        """score 80, budget ok -> executed True, record_spend called."""
        decision = engine.evaluate_and_buy("BTCGBP")
        assert decision.executed is True
        mock_budget.record_spend.assert_called_once()

    def test_spend_not_recorded_if_order_fails(
        self,
        engine: BuyEngine,
        mock_order_mgr: MagicMock,
        mock_budget: MagicMock,
    ) -> None:
        """order_manager raises OrderError -> record_spend NOT called."""
        mock_order_mgr.place_limit_buy.side_effect = OrderError("API error")
        decision = engine.evaluate_and_buy("BTCGBP")
        assert decision.executed is False
        mock_budget.record_spend.assert_not_called()

    def test_evaluate_all_returns_all_assets(
        self, engine: BuyEngine
    ) -> None:
        """Three enabled assets -> three BuyDecisions."""
        decisions = engine.evaluate_all()
        assert len(decisions) == 3

    def test_evaluate_all_never_raises(
        self,
        engine: BuyEngine,
        mock_scorer: MagicMock,
    ) -> None:
        """order_manager raises for one asset -> all assets still evaluated."""
        call_count = 0

        def side_effect(symbol, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Unexpected error")
            return CompositeScore(score=80.0, should_buy=True)

        mock_scorer.compute.side_effect = side_effect
        decisions = engine.evaluate_all()
        assert len(decisions) == 3
        # At least one should have errored
        error_decisions = [d for d in decisions if "error" in d.reason]
        assert len(error_decisions) >= 1

    def test_kill_switch_blocks_all_buys(
        self,
        engine: BuyEngine,
        mock_order_mgr: MagicMock,
    ) -> None:
        """kill switch active -> all decisions executed=False."""
        engine.set_kill_switch(True)
        decision = engine.evaluate_and_buy("BTCGBP")
        assert decision.executed is False
        assert decision.reason == "kill_switch_active"
        mock_order_mgr.place_limit_buy.assert_not_called()
