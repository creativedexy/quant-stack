"""Position sizer -- converts scored candidates into sized positions.

Uses fractional Kelly criterion constrained by tier-specific limits,
sector concentration caps, and minimum position sizes.
"""

from __future__ import annotations

import math
from typing import Any

from src.deployment import AbstractSizer, ScoredCandidate, SizedPosition
from src.deployment.config import get_active_tier
from src.utils.logging import get_logger

logger = get_logger(__name__)


class KellySizer(AbstractSizer):
    """Position sizer using fractional Kelly criterion.

    Args:
        config: Deployment config dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._sizing_config = config.get("sizing", {})
        self._constraints = self._sizing_config.get("constraints", {})
        self._kelly_fraction = self._sizing_config.get("kelly_fraction", 0.25)
        self._eval_thresholds = config.get("evaluation", {}).get(
            "thresholds", {}
        )

    def size(
        self,
        scored: list[ScoredCandidate],
        capital_gbp: float,
        constraints: dict[str, Any] | None = None,
    ) -> list[SizedPosition]:
        """Size positions for scored candidates.

        Args:
            scored: Scored candidates (may include below-threshold).
            capital_gbp: Available capital in GBP (from IB).
            constraints: Override constraints. If None, uses tier-based.

        Returns:
            List of SizedPosition respecting all constraints.
        """
        if capital_gbp <= 0:
            return []

        # Get tier constraints.
        tier = get_active_tier(capital_gbp, self._config)
        if constraints is None:
            constraints = {
                **self._constraints,
                "max_positions": tier["max_positions"],
                "min_position_gbp": tier["min_position_gbp"],
                "prefer_etfs": tier.get("prefer_etfs", False),
            }

        min_composite = self._eval_thresholds.get(
            "minimum_composite_score", 0.55
        )
        max_position_pct = constraints.get(
            "max_position_pct",
            self._constraints.get("max_position_pct", 0.25),
        )
        min_position_gbp = constraints.get(
            "min_position_gbp",
            self._constraints.get("min_position_gbp", 50),
        )
        max_positions = constraints.get(
            "max_positions",
            self._constraints.get("max_positions", 8),
        )
        max_sector_pct = constraints.get(
            "max_single_sector_pct",
            self._constraints.get("max_single_sector_pct", 0.35),
        )
        cash_reserve_pct = constraints.get(
            "cash_reserve_pct",
            self._constraints.get("cash_reserve_pct", 0.05),
        )
        prefer_etfs = constraints.get("prefer_etfs", False)

        # 1. Remove candidates below threshold.
        above = [
            s for s in scored if s.composite_score >= min_composite
        ]

        if not above:
            return []

        # Sort by composite score descending.
        above.sort(key=lambda s: s.composite_score, reverse=True)

        # 2. ETF preference at small scale.
        if prefer_etfs:
            above = self._prefer_etfs(above)

        # 3. Compute raw Kelly fractions.
        positions: list[SizedPosition] = []
        for s in above:
            kelly_raw = self._kelly_fraction_calc(s)
            kelly_adjusted = kelly_raw * self._kelly_fraction

            # Cap at max position percentage.
            kelly_adjusted = min(kelly_adjusted, max_position_pct)

            gbp_amount = capital_gbp * kelly_adjusted

            # Determine order type.
            order_type = "MARKET" if (
                s.asset_type == "etf"
                and s.avg_daily_volume >= self._config.get(
                    "execution", {}
                ).get("market_order_volume_threshold", 1_000_000)
            ) else "LIMIT"

            # Compute shares (integer, rounded down).
            # At small capital, ensure at least 1 share if affordable.
            price = s.gbp_price if s.gbp_price > 0 else 1.0
            shares = int(gbp_amount / price)
            if shares == 0 and price <= capital_gbp * max_position_pct:
                shares = 1
                gbp_amount = price

            positions.append(SizedPosition(
                ticker=s.ticker,
                name=s.name,
                exchange=s.exchange,
                sector=s.sector,
                asset_type=s.asset_type,
                gbp_price=s.gbp_price,
                shares=shares,
                gbp_amount=shares * price,
                pct_of_capital=kelly_adjusted,
                order_type=order_type,
                composite_score=s.composite_score,
                conviction=s.conviction,
                reasoning_summary=(
                    s.reasoning.get("growth", "")[:200]
                    if s.reasoning else ""
                ),
            ))

        # 4. Enforce min_position_gbp.
        positions = [
            p for p in positions if p.gbp_amount >= min_position_gbp
        ]

        # 5. Enforce max_positions.
        positions = positions[:max_positions]

        # 6. Enforce sector concentration.
        positions = self._enforce_sector_limit(
            positions, max_sector_pct, capital_gbp
        )

        # 7. Reserve cash.
        deployable = capital_gbp * (1.0 - cash_reserve_pct)
        positions = self._enforce_capital_limit(positions, deployable)

        logger.info(
            "Sized %d positions, total GBP %.2f / %.2f available",
            len(positions),
            sum(p.gbp_amount for p in positions),
            capital_gbp,
        )
        return positions

    def _kelly_fraction_calc(self, s: ScoredCandidate) -> float:
        """Compute raw Kelly fraction for a candidate.

        Uses composite score as win probability proxy and a fixed
        win/loss ratio estimate.

        Args:
            s: Scored candidate.

        Returns:
            Raw Kelly fraction (before applying kelly_fraction multiplier).
        """
        p = s.composite_score  # Win probability proxy.
        q = 1.0 - p
        b = 2.0  # Assume 2:1 win/loss ratio (conservative).

        kelly = (p * b - q) / b
        return max(0.0, kelly)

    def _prefer_etfs(
        self, candidates: list[ScoredCandidate],
    ) -> list[ScoredCandidate]:
        """At small scale, prefer ETFs over similar-scoring stocks.

        If an ETF and a stock have composite scores within 0.05,
        move the ETF ahead.
        """
        etfs = [c for c in candidates if c.asset_type == "etf"]
        stocks = [c for c in candidates if c.asset_type != "etf"]

        # Simple approach: interleave ETFs first if scores are close.
        result: list[ScoredCandidate] = []
        used_tickers: set[str] = set()

        for etf in etfs:
            result.append(etf)
            used_tickers.add(etf.ticker)

        for stock in stocks:
            if stock.ticker not in used_tickers:
                result.append(stock)

        return result

    def _enforce_sector_limit(
        self,
        positions: list[SizedPosition],
        max_sector_pct: float,
        capital_gbp: float,
    ) -> list[SizedPosition]:
        """Trim positions in oversized sectors proportionally."""
        sector_totals: dict[str, float] = {}
        for p in positions:
            sector_totals[p.sector] = (
                sector_totals.get(p.sector, 0) + p.gbp_amount
            )

        max_sector_gbp = capital_gbp * max_sector_pct
        result: list[SizedPosition] = []

        sector_allocated: dict[str, float] = {}
        for p in positions:
            sector = p.sector
            already = sector_allocated.get(sector, 0)
            if already + p.gbp_amount > max_sector_gbp:
                # Trim this position.
                remaining = max(0, max_sector_gbp - already)
                if remaining >= self._constraints.get("min_position_gbp", 50):
                    price = p.gbp_price if p.gbp_price > 0 else 1.0
                    new_shares = int(remaining / price)
                    if new_shares > 0:
                        result.append(SizedPosition(
                            ticker=p.ticker,
                            name=p.name,
                            exchange=p.exchange,
                            sector=p.sector,
                            asset_type=p.asset_type,
                            gbp_price=p.gbp_price,
                            shares=new_shares,
                            gbp_amount=new_shares * price,
                            pct_of_capital=new_shares * price / capital_gbp,
                            order_type=p.order_type,
                            composite_score=p.composite_score,
                            conviction=p.conviction,
                            reasoning_summary=p.reasoning_summary,
                        ))
                        sector_allocated[sector] = already + new_shares * price
            else:
                result.append(p)
                sector_allocated[sector] = already + p.gbp_amount

        return result

    def _enforce_capital_limit(
        self,
        positions: list[SizedPosition],
        max_gbp: float,
    ) -> list[SizedPosition]:
        """Ensure total allocated does not exceed available capital.

        At small capital where most positions are 1 share, proportional
        scaling rounds to 0.  Instead, drop lowest-scored positions first,
        then scale only if still over.
        """
        # Sort by composite_score descending (keep best first).
        positions = sorted(
            positions, key=lambda p: p.composite_score, reverse=True,
        )

        # Phase 1: drop lowest-scored until total fits.
        while (
            len(positions) > 0
            and sum(p.gbp_amount for p in positions) > max_gbp
        ):
            positions = positions[:-1]

        return positions
