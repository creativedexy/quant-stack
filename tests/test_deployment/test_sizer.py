"""Tests for src/deployment/sizer.py."""

from __future__ import annotations

import pytest

from src.deployment import ScoredCandidate, SizedPosition
from src.deployment.sizer import KellySizer


@pytest.fixture
def sizer(deployment_config) -> KellySizer:
    """KellySizer with test config."""
    return KellySizer(config=deployment_config)


@pytest.fixture
def scored_candidates() -> list[ScoredCandidate]:
    """High-scoring candidates for sizing tests."""
    base = {
        "exchange": "NASDAQ",
        "asset_type": "stock",
        "gbp_price": 100.0,
        "avg_daily_volume": 500_000,
        "currency": "USD",
        "reasoning": {"growth": "Strong growth.", "diversification": "Good."},
        "risks": [],
        "conviction": "HIGH",
    }
    return [
        ScoredCandidate(
            ticker="AAAA",
            name="Alpha Corp",
            sector="Technology",
            market_cap_usd=100e9,
            source_universe="us_growth",
            scores={"growth": 0.85, "diversification": 0.70, "risk_adjusted": 0.65, "income": 0.30, "composite": 0.72},
            **base,
        ),
        ScoredCandidate(
            ticker="BBBB",
            name="Beta Inc",
            sector="Healthcare",
            market_cap_usd=80e9,
            source_universe="us_growth",
            scores={"growth": 0.75, "diversification": 0.80, "risk_adjusted": 0.60, "income": 0.40, "composite": 0.68},
            **base,
        ),
        ScoredCandidate(
            ticker="CCCC",
            name="Gamma Ltd",
            sector="Energy",
            market_cap_usd=50e9,
            source_universe="contrarian_value",
            scores={"growth": 0.60, "diversification": 0.75, "risk_adjusted": 0.70, "income": 0.55, "composite": 0.64},
            **base,
        ),
        ScoredCandidate(
            ticker="DDDD",
            name="Delta plc",
            sector="Technology",
            market_cap_usd=30e9,
            source_universe="us_growth",
            scores={"growth": 0.70, "diversification": 0.60, "risk_adjusted": 0.55, "income": 0.35, "composite": 0.60},
            **base,
        ),
    ]


# ------------------------------------------------------------------
# Basic sizing behaviour
# ------------------------------------------------------------------


class TestKellySizerBasic:
    """Basic sizing tests."""

    def test_returns_sized_positions(self, sizer, scored_candidates):
        """Sizer returns SizedPosition objects."""
        result = sizer.size(scored_candidates, capital_gbp=5000.0)
        assert len(result) > 0
        for p in result:
            assert isinstance(p, SizedPosition)

    def test_zero_capital_returns_empty(self, sizer, scored_candidates):
        """Zero capital returns no positions."""
        result = sizer.size(scored_candidates, capital_gbp=0.0)
        assert result == []

    def test_negative_capital_returns_empty(self, sizer, scored_candidates):
        """Negative capital returns no positions."""
        result = sizer.size(scored_candidates, capital_gbp=-100.0)
        assert result == []

    def test_all_below_threshold_returns_empty(self, sizer):
        """Candidates below minimum composite score are excluded."""
        low_scores = [
            ScoredCandidate(
                ticker="LOW",
                name="Low Score",
                exchange="NYSE",
                sector="Tech",
                market_cap_usd=10e9,
                source_universe="us_growth",
                asset_type="stock",
                gbp_price=50.0,
                avg_daily_volume=100_000,
                currency="USD",
                scores={"growth": 0.3, "diversification": 0.3, "risk_adjusted": 0.3, "income": 0.3, "composite": 0.30},
                reasoning={},
                risks=[],
                conviction="LOW",
            ),
        ]
        result = sizer.size(low_scores, capital_gbp=5000.0)
        assert result == []


# ------------------------------------------------------------------
# Constraint enforcement
# ------------------------------------------------------------------


class TestKellySizerConstraints:
    """Constraint enforcement tests."""

    def test_total_does_not_exceed_capital(self, sizer, scored_candidates):
        """Total allocated GBP does not exceed available capital."""
        capital = 5000.0
        result = sizer.size(scored_candidates, capital_gbp=capital)
        total = sum(p.gbp_amount for p in result)
        assert total <= capital

    def test_no_position_exceeds_max_pct(self, sizer, scored_candidates):
        """No single position exceeds max_position_pct of capital."""
        capital = 5000.0
        max_pct = sizer._constraints.get("max_position_pct", 0.25)
        result = sizer.size(scored_candidates, capital_gbp=capital)
        for p in result:
            assert p.gbp_amount <= capital * max_pct + 1.0  # +1 for rounding.

    def test_min_position_gbp_enforced(self, sizer, scored_candidates):
        """All positions meet minimum GBP size."""
        capital = 5000.0
        result = sizer.size(scored_candidates, capital_gbp=capital)
        # The min from tier is used; at 5000 GBP it should be tier_2.
        for p in result:
            assert p.gbp_amount >= 25  # Tier 2 min is 25.

    def test_max_positions_enforced(self, sizer, scored_candidates):
        """Number of positions does not exceed max_positions for tier."""
        capital = 500.0  # Tier 1: max 5 positions.
        result = sizer.size(scored_candidates, capital_gbp=capital)
        assert len(result) <= 5

    def test_sector_concentration_limit(self, sizer):
        """Sector concentration is capped."""
        # Create 3 candidates all in same sector with high scores.
        same_sector = []
        for i, ticker in enumerate(["S1", "S2", "S3"]):
            same_sector.append(ScoredCandidate(
                ticker=ticker,
                name=f"Same Sector {i}",
                exchange="NYSE",
                sector="Technology",
                market_cap_usd=50e9,
                source_universe="us_growth",
                asset_type="stock",
                gbp_price=50.0,
                avg_daily_volume=500_000,
                currency="USD",
                scores={"growth": 0.85, "diversification": 0.70, "risk_adjusted": 0.65, "income": 0.30, "composite": 0.72},
                reasoning={"growth": "x"},
                risks=[],
                conviction="HIGH",
            ))
        capital = 5000.0
        result = sizer.size(same_sector, capital_gbp=capital)
        tech_total = sum(p.gbp_amount for p in result if p.sector == "Technology")
        max_sector_pct = sizer._constraints.get("max_single_sector_pct", 0.35)
        assert tech_total <= capital * max_sector_pct + 1.0  # +1 for rounding.

    def test_cash_reserve_maintained(self, sizer, scored_candidates):
        """Cash reserve percentage is maintained."""
        capital = 5000.0
        cash_pct = sizer._constraints.get("cash_reserve_pct", 0.05)
        result = sizer.size(scored_candidates, capital_gbp=capital)
        total = sum(p.gbp_amount for p in result)
        assert total <= capital * (1.0 - cash_pct) + 1.0  # +1 for rounding.


# ------------------------------------------------------------------
# Kelly criterion & order type
# ------------------------------------------------------------------


class TestKellySizerKelly:
    """Kelly criterion and special behaviour tests."""

    def test_kelly_fraction_positive_for_good_score(self, sizer):
        """Kelly fraction is positive for a high-scoring candidate."""
        candidate = ScoredCandidate(
            ticker="GOOD",
            name="Good Co",
            exchange="NYSE",
            sector="Tech",
            market_cap_usd=50e9,
            source_universe="us_growth",
            asset_type="stock",
            gbp_price=100.0,
            avg_daily_volume=500_000,
            currency="USD",
            scores={"growth": 0.80, "diversification": 0.70, "risk_adjusted": 0.60, "income": 0.40, "composite": 0.70},
            reasoning={},
            risks=[],
            conviction="HIGH",
        )
        raw = sizer._kelly_fraction_calc(candidate)
        assert raw > 0

    def test_kelly_fraction_zero_for_low_score(self, sizer):
        """Kelly fraction is zero for a low-scoring candidate."""
        candidate = ScoredCandidate(
            ticker="BAD",
            name="Bad Co",
            exchange="NYSE",
            sector="Tech",
            market_cap_usd=50e9,
            source_universe="us_growth",
            asset_type="stock",
            gbp_price=100.0,
            avg_daily_volume=500_000,
            currency="USD",
            scores={"growth": 0.20, "diversification": 0.20, "risk_adjusted": 0.20, "income": 0.20, "composite": 0.20},
            reasoning={},
            risks=[],
            conviction="LOW",
        )
        raw = sizer._kelly_fraction_calc(candidate)
        assert raw == 0.0

    def test_shares_are_integers(self, sizer, scored_candidates):
        """All share counts are integers (rounded down)."""
        result = sizer.size(scored_candidates, capital_gbp=5000.0)
        for p in result:
            assert isinstance(p.shares, int)
            assert p.shares > 0

    def test_etf_preference_at_small_scale(self, sizer):
        """ETFs are preferred over stocks at small capital scale."""
        etf = ScoredCandidate(
            ticker="VTI",
            name="Total Market ETF",
            exchange="NYSE",
            sector="Diversified",
            market_cap_usd=300e9,
            source_universe="sector_etfs",
            asset_type="etf",
            gbp_price=50.0,
            avg_daily_volume=2_000_000,
            currency="USD",
            scores={"growth": 0.70, "diversification": 0.80, "risk_adjusted": 0.70, "income": 0.40, "composite": 0.70},
            reasoning={"growth": "x"},
            risks=[],
            conviction="MEDIUM",
        )
        stock = ScoredCandidate(
            ticker="TINY",
            name="Tiny Stock",
            exchange="NYSE",
            sector="Diversified",
            market_cap_usd=5e9,
            source_universe="us_growth",
            asset_type="stock",
            gbp_price=50.0,
            avg_daily_volume=200_000,
            currency="USD",
            scores={"growth": 0.72, "diversification": 0.78, "risk_adjusted": 0.68, "income": 0.38, "composite": 0.69},
            reasoning={"growth": "x"},
            risks=[],
            conviction="MEDIUM",
        )
        # At tier 1 (prefer_etfs=True), ETF should come first.
        result = sizer.size(
            [stock, etf],
            capital_gbp=400.0,  # Tier 1.
        )
        if len(result) >= 2:
            # ETF should be first.
            assert result[0].asset_type == "etf"

    def test_market_order_for_liquid_etf(self, sizer):
        """Liquid ETFs get MARKET order type."""
        etf = ScoredCandidate(
            ticker="SPY",
            name="S&P 500 ETF",
            exchange="NYSE",
            sector="Diversified",
            market_cap_usd=400e9,
            source_universe="sector_etfs",
            asset_type="etf",
            gbp_price=50.0,
            avg_daily_volume=50_000_000,  # Very liquid.
            currency="USD",
            scores={"growth": 0.75, "diversification": 0.85, "risk_adjusted": 0.70, "income": 0.35, "composite": 0.72},
            reasoning={"growth": "x"},
            risks=[],
            conviction="HIGH",
        )
        result = sizer.size([etf], capital_gbp=5000.0)
        if result:
            assert result[0].order_type == "MARKET"
