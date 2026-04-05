"""Buy engine that connects signals, budget, and order placement.

This is the integration hub -- it evaluates composite scores and
makes buy/no-buy decisions for each asset.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.execution.budget_tracker import BudgetTracker
from src.execution.order_manager import BinanceOrderManager, OrderResult
from src.signals.scorer import CompositeScore, SignalScorer
from src.utils.exceptions import OrderError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BuyDecision:
    """Result of an evaluate-and-buy cycle for one asset.

    Attributes:
        asset: Trading pair symbol.
        executed: True if a buy order was placed.
        score: CompositeScore from the signal scorer.
        order_result: OrderResult if executed, else None.
        reason: Human-readable explanation of the decision.
    """

    asset: str
    executed: bool
    score: CompositeScore
    order_result: OrderResult | None
    reason: str


class BuyEngine:
    """Evaluates signals and places orders when conditions are met.

    Ordering of operations in evaluate_and_buy is strict:
    1. Score -> 2. Threshold check -> 3. Budget check -> 4. Price ->
    5. Risk validation -> 6. Order placement.

    Args:
        scorer: SignalScorer instance for computing composite scores.
        budget_tracker: BudgetTracker for checking/recording spend.
        order_manager: BinanceOrderManager for placing orders.
        price_feed: BinancePriceFeed for getting current prices.
        config: Full settings dict from config_loader.
    """

    def __init__(
        self,
        scorer: SignalScorer,
        budget_tracker: BudgetTracker,
        order_manager: BinanceOrderManager,
        price_feed: object,
        config: dict,
    ) -> None:
        self._scorer = scorer
        self._budget = budget_tracker
        self._orders = order_manager
        self._price_feed = price_feed
        self._config = config
        scoring = config.get("scoring", {})
        self._buy_threshold = scoring.get("buy_threshold", 65)
        self._assets = config.get("assets", {})
        self._kill_switch_active = False

    def set_kill_switch(self, active: bool) -> None:
        """Activate or deactivate the kill switch.

        Args:
            active: True to block all buys.
        """
        self._kill_switch_active = active
        if active:
            logger.warning("kill_switch_activated")

    def evaluate_and_buy(self, asset: str) -> BuyDecision:
        """Full pipeline: score -> threshold -> budget -> price -> order.

        Args:
            asset: Trading pair symbol (e.g. 'BTCGBP').

        Returns:
            BuyDecision with execution details.
        """
        # Step 0: Kill switch
        if self._kill_switch_active:
            score = CompositeScore(score=0.0)
            return BuyDecision(
                asset=asset,
                executed=False,
                score=score,
                order_result=None,
                reason="kill_switch_active",
            )

        # Step 1: Compute composite score
        score = self._scorer.compute(asset)

        # Step 2: Threshold check
        if score.score < self._buy_threshold:
            logger.info(
                "score_below_threshold",
                asset=asset,
                score=score.score,
                threshold=self._buy_threshold,
            )
            return BuyDecision(
                asset=asset,
                executed=False,
                score=score,
                order_result=None,
                reason=f"score {score.score:.1f} below threshold {self._buy_threshold}",
            )

        # Step 3: Budget check
        asset_cfg = self._assets.get(asset, {})
        min_order = asset_cfg.get("min_order_gbp", 5.0)
        remaining = self._budget.get_remaining(asset)

        if remaining < min_order:
            logger.info(
                "budget_exhausted",
                asset=asset,
                remaining=remaining,
                min_order=min_order,
            )
            return BuyDecision(
                asset=asset,
                executed=False,
                score=score,
                order_result=None,
                reason=(
                    f"budget exhausted: {remaining:.2f} GBP remaining,"
                    f" min order {min_order:.2f}"
                ),
            )

        # Use remaining budget as spend amount (capped at weekly allocation remaining)
        amount_gbp = min(remaining, asset_cfg.get("weekly_allocation_gbp", remaining))

        # Step 4: Get current midprice
        midprice = self._price_feed.get_midprice(asset)

        # Step 5 & 6: Place order (risk validation inside order_manager)
        try:
            order_result = self._orders.place_limit_buy(
                asset=asset,
                amount_gbp=amount_gbp,
                limit_price=midprice,
            )
        except OrderError as exc:
            logger.error("order_placement_failed", asset=asset, error=str(exc))
            return BuyDecision(
                asset=asset,
                executed=False,
                score=score,
                order_result=None,
                reason=f"order failed: {exc}",
            )

        # Only record spend AFTER successful order
        self._budget.record_spend(asset, amount_gbp)

        logger.info(
            "buy_executed",
            asset=asset,
            amount_gbp=amount_gbp,
            score=score.score,
            order_id=order_result.order_id,
        )

        return BuyDecision(
            asset=asset,
            executed=True,
            score=score,
            order_result=order_result,
            reason="buy executed",
        )

    def evaluate_all(self) -> list[BuyDecision]:
        """Run evaluate_and_buy for all enabled assets.

        Never raises -- returns BuyDecision with executed=False for
        any asset that errors.

        Returns:
            List of BuyDecision objects, one per enabled asset.
        """
        decisions: list[BuyDecision] = []

        for asset, cfg in self._assets.items():
            if not cfg.get("enabled", True):
                continue

            try:
                decision = self.evaluate_and_buy(asset)
            except Exception as exc:
                logger.error(
                    "evaluate_and_buy_error",
                    asset=asset,
                    error=str(exc),
                )
                decision = BuyDecision(
                    asset=asset,
                    executed=False,
                    score=CompositeScore(score=0.0),
                    order_result=None,
                    reason=f"error: {exc}",
                )
            decisions.append(decision)

        return decisions
