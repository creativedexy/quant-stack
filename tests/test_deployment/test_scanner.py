"""Tests for src/deployment/scanner.py."""

from __future__ import annotations

import pytest

from src.deployment import Candidate
from src.deployment.scanner import ScannerService, SyntheticScanner


# ------------------------------------------------------------------
# SyntheticScanner
# ------------------------------------------------------------------


class TestSyntheticScanner:
    """Tests for the offline synthetic scanner."""

    def test_returns_15_candidates(self):
        """Synthetic scanner returns exactly 15 candidates."""
        scanner = SyntheticScanner()
        candidates = scanner.scan()
        assert len(candidates) == 15

    def test_candidates_are_candidate_objects(self):
        """All items are Candidate dataclass instances."""
        scanner = SyntheticScanner()
        candidates = scanner.scan()
        for c in candidates:
            assert isinstance(c, Candidate)

    def test_three_universes_represented(self):
        """Candidates span all three source universes."""
        scanner = SyntheticScanner()
        candidates = scanner.scan()
        universes = {c.source_universe for c in candidates}
        assert universes == {"us_growth", "sector_etfs", "contrarian_value"}

    def test_five_per_universe(self):
        """Five candidates from each universe."""
        scanner = SyntheticScanner()
        candidates = scanner.scan()
        for universe in ["us_growth", "sector_etfs", "contrarian_value"]:
            count = sum(
                1 for c in candidates if c.source_universe == universe
            )
            assert count == 5, f"{universe} has {count} candidates"

    def test_exclusions_applied(self):
        """Excluded tickers are removed from results."""
        scanner = SyntheticScanner()
        candidates = scanner.scan(exclusions=["NVDA", "SMH"])
        tickers = {c.ticker for c in candidates}
        assert "NVDA" not in tickers
        assert "SMH" not in tickers
        assert len(candidates) == 13

    def test_exclusions_case_insensitive(self):
        """Exclusion matching is case-insensitive."""
        scanner = SyntheticScanner()
        candidates = scanner.scan(exclusions=["nvda"])
        tickers = {c.ticker for c in candidates}
        assert "NVDA" not in tickers

    def test_all_fields_populated(self):
        """All required Candidate fields are populated."""
        scanner = SyntheticScanner()
        candidates = scanner.scan()
        for c in candidates:
            assert c.ticker
            assert c.name
            assert c.exchange
            assert c.sector
            assert c.market_cap_usd > 0
            assert c.source_universe
            assert c.avg_daily_volume > 0
            assert c.currency
            assert c.asset_type in ("stock", "etf")

    def test_etfs_have_correct_asset_type(self):
        """Sector ETF candidates are marked as asset_type='etf'."""
        scanner = SyntheticScanner()
        candidates = scanner.scan()
        etfs = [c for c in candidates if c.source_universe == "sector_etfs"]
        for c in etfs:
            assert c.asset_type == "etf"


# ------------------------------------------------------------------
# ScannerService (with mocked yfinance)
# ------------------------------------------------------------------


class TestScannerService:
    """Tests for the real scanner service (mocked network)."""

    def test_portfolio_exclusions_applied(self, deployment_config):
        """Tickers held in portfolio are excluded."""

        class MockPortfolioService:
            def get_positions(self):
                return [{"ticker": "NVDA"}, {"ticker": "SMH"}]

        class TestableScanner(ScannerService):
            def _scan_universe(self, name, config):
                return SyntheticScanner().scan()[:5]

        scanner = TestableScanner(
            config=deployment_config,
            portfolio_service=MockPortfolioService(),
        )
        candidates = scanner.scan()
        tickers = {c.ticker for c in candidates}
        assert "NVDA" not in tickers
        assert "SMH" not in tickers

    def test_memory_cooldown_exclusions(self, deployment_config):
        """Tickers in memory cooldown are excluded."""
        from unittest.mock import MagicMock

        mock_memory = MagicMock()
        mock_memory.get_rejected_tickers.return_value = ["RIO"]

        class TestableScanner(ScannerService):
            def _scan_universe(self, name, config):
                return SyntheticScanner().scan()[:5]

        scanner = TestableScanner(
            config=deployment_config,
            memory=mock_memory,
        )
        candidates = scanner.scan()
        tickers = {c.ticker for c in candidates}
        assert "RIO" not in tickers

    def test_deduplication_across_universes(self, deployment_config):
        """Duplicate tickers across universes are deduplicated."""
        dupe = Candidate(
            "SMH", "Dupe", "NASDAQ", "Tech", 1e9, "us_growth", 1e6
        )

        class TestableScanner(ScannerService):
            def _scan_universe(self, name, config):
                if name == "us_growth":
                    return [dupe]
                elif name == "sector_etfs":
                    return SyntheticScanner().scan()[:3]
                return []

        scanner = TestableScanner(config=deployment_config)
        candidates = scanner.scan(exclusions=[])
        smh_count = sum(1 for c in candidates if c.ticker == "SMH")
        assert smh_count == 1

    def test_liquidity_filter(self, deployment_config):
        """Candidates below minimum volume are removed."""
        low_vol = Candidate(
            "ILLIQUID", "Illiquid Co", "NYSE", "Tech", 10e9,
            "us_growth", 100,  # Very low volume.
        )

        class TestableScanner(ScannerService):
            def _scan_universe(self, name, config):
                if name == "us_growth":
                    return [low_vol]
                return []

        scanner = TestableScanner(config=deployment_config)
        candidates = scanner.scan(exclusions=[])
        tickers = {c.ticker for c in candidates}
        assert "ILLIQUID" not in tickers

    def test_empty_universe_returns_empty(self, deployment_config):
        """An empty universe scan returns no candidates."""

        class TestableScanner(ScannerService):
            def _scan_universe(self, name, config):
                return []

        scanner = TestableScanner(config=deployment_config)
        candidates = scanner.scan(exclusions=[])
        assert candidates == []

    def test_failed_universe_does_not_crash(self, deployment_config):
        """A failing universe scan is caught and logged."""

        class TestableScanner(ScannerService):
            def _scan_universe(self, name, config):
                if name == "us_growth":
                    raise RuntimeError("API down")
                return SyntheticScanner().scan()[:5]

        scanner = TestableScanner(config=deployment_config)
        # Should not raise.
        candidates = scanner.scan(exclusions=[])
        # Should still have results from other universes.
        assert len(candidates) > 0
