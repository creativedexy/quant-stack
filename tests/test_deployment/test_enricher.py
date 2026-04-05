"""Tests for src/deployment/enricher.py."""

from __future__ import annotations

import pytest

from src.deployment import Candidate, EnrichedCandidate
from src.deployment.enricher import SyntheticEnricher


# ------------------------------------------------------------------
# SyntheticEnricher
# ------------------------------------------------------------------


class TestSyntheticEnricher:
    """Tests for the offline synthetic enricher."""

    def test_enriches_all_candidates(self, synthetic_candidates):
        """All input candidates are enriched (no drops)."""
        enricher = SyntheticEnricher()
        result = enricher.enrich(synthetic_candidates)
        assert len(result) == len(synthetic_candidates)

    def test_returns_enriched_candidate_objects(self, synthetic_candidates):
        """All items are EnrichedCandidate dataclass instances."""
        enricher = SyntheticEnricher()
        result = enricher.enrich(synthetic_candidates)
        for ec in result:
            assert isinstance(ec, EnrichedCandidate)

    def test_fundamentals_populated(self, synthetic_candidates):
        """Enriched candidates have non-empty fundamentals dicts."""
        enricher = SyntheticEnricher()
        result = enricher.enrich(synthetic_candidates)
        for ec in result:
            assert ec.fundamentals
            assert "pe_ratio" in ec.fundamentals
            assert "dividend_yield" in ec.fundamentals
            assert ec.fundamentals["pe_ratio"] is not None

    def test_technicals_populated(self, synthetic_candidates):
        """Enriched candidates have non-empty technicals dicts."""
        enricher = SyntheticEnricher()
        result = enricher.enrich(synthetic_candidates)
        for ec in result:
            assert ec.technicals
            assert "rsi_14" in ec.technicals
            assert "macd_line" in ec.technicals

    def test_rsi_bounded(self, synthetic_candidates):
        """RSI values are within the valid [0, 100] range."""
        enricher = SyntheticEnricher()
        result = enricher.enrich(synthetic_candidates)
        for ec in result:
            rsi = ec.technicals.get("rsi_14", 50)
            assert 0 <= rsi <= 100, f"{ec.ticker} RSI={rsi}"

    def test_gbp_price_positive(self, synthetic_candidates):
        """GBP prices are positive for all enriched candidates."""
        enricher = SyntheticEnricher()
        result = enricher.enrich(synthetic_candidates)
        for ec in result:
            assert ec.gbp_price > 0, f"{ec.ticker} gbp_price={ec.gbp_price}"

    def test_news_sentiment_present(self, synthetic_candidates):
        """News sentiment dict is present with required keys."""
        enricher = SyntheticEnricher()
        result = enricher.enrich(synthetic_candidates)
        for ec in result:
            assert "score" in ec.news_sentiment
            assert "available" in ec.news_sentiment
            score = ec.news_sentiment["score"]
            assert 0 <= score <= 1

    def test_identity_fields_preserved(self, synthetic_candidates):
        """Ticker, name, sector, etc. are preserved from Candidate."""
        enricher = SyntheticEnricher()
        result = enricher.enrich(synthetic_candidates)
        for ec, c in zip(result, synthetic_candidates):
            assert ec.ticker == c.ticker
            assert ec.name == c.name
            assert ec.sector == c.sector
            assert ec.exchange == c.exchange
            assert ec.source_universe == c.source_universe

    def test_etf_asset_type_preserved(self, synthetic_candidates):
        """ETF asset type is carried through enrichment."""
        enricher = SyntheticEnricher()
        result = enricher.enrich(synthetic_candidates)
        for ec, c in zip(result, synthetic_candidates):
            assert ec.asset_type == c.asset_type

    def test_different_tickers_get_different_data(self):
        """Enrichment produces distinct data per ticker (seeded)."""
        candidates = [
            Candidate("AAA", "Alpha", "NYSE", "Tech", 10e9, "test", 1e6),
            Candidate("ZZZ", "Omega", "NYSE", "Energy", 5e9, "test", 500_000),
        ]
        enricher = SyntheticEnricher()
        result = enricher.enrich(candidates)
        assert result[0].fundamentals["pe_ratio"] != result[1].fundamentals["pe_ratio"]
