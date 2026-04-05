"""Universe scanner -- discovers investment candidates across multiple universes.

Scans US growth stocks, sector ETFs, and contrarian value picks.
Applies dynamic exclusions from portfolio holdings, memory cooldowns,
and structural config exclusions. Deduplicates across universes.
"""

from __future__ import annotations

import time
from typing import Any

from src.deployment import AbstractScanner, Candidate
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ScannerService(AbstractScanner):
    """Scans configured universes and returns filtered candidates.

    Args:
        config: Deployment config dict.
        memory: MemoryManager instance (for cooldown lookups).
        portfolio_service: Optional PortfolioService for held tickers.
    """

    def __init__(
        self,
        config: dict[str, Any],
        memory: Any | None = None,
        portfolio_service: Any | None = None,
    ) -> None:
        self._config = config
        self._memory = memory
        self._portfolio_service = portfolio_service
        self._scanner_config = config.get("scanner", {})

    def scan(self, exclusions: list[str] | None = None) -> list[Candidate]:
        """Scan all configured universes with exclusions applied.

        Args:
            exclusions: Additional tickers to exclude. If None, builds
                the exclusion list dynamically from portfolio + memory + config.

        Returns:
            Deduplicated, filtered list of Candidate objects.
        """
        if exclusions is None:
            exclusions = self._build_exclusion_list()

        exclusion_set = {t.upper() for t in exclusions}

        all_candidates: list[Candidate] = []
        universes = self._scanner_config.get("universes", {})

        for universe_name, universe_config in universes.items():
            try:
                candidates = self._scan_universe(
                    universe_name, universe_config
                )
                logger.info(
                    "Scanned %s: %d raw candidates",
                    universe_name,
                    len(candidates),
                )
                all_candidates.extend(candidates)
            except Exception as e:
                logger.warning(
                    "Failed to scan universe %s: %s", universe_name, e
                )

        # Apply exclusions.
        filtered = self._apply_exclusions(all_candidates, exclusion_set)

        # Apply liquidity filter.
        filtered = self._apply_liquidity_filter(filtered)

        # Deduplicate (keep first occurrence).
        filtered = self._deduplicate(filtered)

        logger.info(
            "Scanner complete: %d candidates after filtering "
            "(excluded %d, pre-filter %d)",
            len(filtered),
            len(all_candidates) - len(filtered),
            len(all_candidates),
        )
        return filtered

    def _build_exclusion_list(self) -> list[str]:
        """Build dynamic exclusions from portfolio + memory + config."""
        exclusions: list[str] = []

        # 1. Current portfolio holdings.
        if self._portfolio_service is not None:
            try:
                positions = self._portfolio_service.get_positions()
                exclusions.extend(p.get("ticker", "") for p in positions)
            except Exception as e:
                logger.warning(
                    "Could not read portfolio positions: %s", e
                )

        # 2. Structural exclusions from config.
        structural = (
            self._scanner_config
            .get("global_exclusions", {})
            .get("structural", [])
        )
        exclusions.extend(structural)

        # 3. Rejected candidates with active cooldown.
        if self._memory is not None:
            try:
                cooldown = self._memory.get_rejected_tickers()
                exclusions.extend(cooldown)
            except Exception as e:
                logger.warning(
                    "Could not read memory cooldowns: %s", e
                )

        return exclusions

    def _scan_universe(
        self,
        name: str,
        config: dict[str, Any],
    ) -> list[Candidate]:
        """Scan a single universe and return raw candidates."""
        if name == "sector_etfs":
            return self._scan_sector_etfs(config)
        elif name == "us_growth":
            return self._scan_us_growth(config)
        elif name == "contrarian_value":
            return self._scan_contrarian_value(config)
        else:
            logger.warning("Unknown universe: %s", name)
            return []

    def _scan_sector_etfs(
        self, config: dict[str, Any],
    ) -> list[Candidate]:
        """Scan sector ETFs from the ticker list in config."""
        tickers = config.get("tickers", [])
        max_candidates = config.get("max_candidates", len(tickers))
        candidates: list[Candidate] = []

        for ticker in tickers[:max_candidates]:
            try:
                info = self._fetch_ticker_info(ticker)
                if info is None:
                    continue

                candidates.append(Candidate(
                    ticker=ticker,
                    name=info.get("shortName", info.get("longName", ticker)),
                    exchange=info.get("exchange", "UNKNOWN"),
                    sector=info.get("sector", "Diversified"),
                    market_cap_usd=float(
                        info.get("totalAssets", 0)
                        or info.get("marketCap", 0)
                        or 0
                    ),
                    source_universe="sector_etfs",
                    avg_daily_volume=float(
                        info.get("averageVolume", 0) or 0
                    ),
                    currency=info.get("currency", "USD"),
                    asset_type="etf",
                ))
            except Exception as e:
                logger.warning("Failed to fetch ETF info for %s: %s", ticker, e)

        return candidates

    def _scan_us_growth(
        self, config: dict[str, Any],
    ) -> list[Candidate]:
        """Scan US growth stocks matching filter criteria."""
        filters = config.get("filter", {})
        max_candidates = config.get("max_candidates", 20)

        # Use a curated seed list of large-cap US growth names.
        # In production, this would come from a screener API.
        seed_tickers = [
            "NVDA", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "ORCL",
            "CRM", "AMD", "NFLX", "ADBE", "NOW", "UBER", "PANW",
            "ABNB", "SNOW", "PLTR", "COIN", "SHOP", "SQ",
            "ASML", "TSM", "INTU", "ISRG", "LRCX",
        ]

        excluded_sectors = {
            s.lower() for s in filters.get("sector_exclude", [])
        }
        min_cap = filters.get("market_cap_min_usd", 0)
        allowed_exchanges = {
            e.upper() for e in filters.get("exchange", [])
        }

        candidates: list[Candidate] = []

        for ticker in seed_tickers:
            if len(candidates) >= max_candidates:
                break

            try:
                info = self._fetch_ticker_info(ticker)
                if info is None:
                    continue

                sector = info.get("sector", "Unknown")
                if sector.lower() in excluded_sectors:
                    continue

                cap = float(info.get("marketCap", 0) or 0)
                if cap < min_cap:
                    continue

                exchange = info.get("exchange", "").upper()
                if allowed_exchanges and not any(
                    ex in exchange for ex in allowed_exchanges
                ):
                    continue

                candidates.append(Candidate(
                    ticker=ticker,
                    name=info.get("shortName", info.get("longName", ticker)),
                    exchange=exchange,
                    sector=sector,
                    market_cap_usd=cap,
                    source_universe="us_growth",
                    avg_daily_volume=float(
                        info.get("averageVolume", 0) or 0
                    ),
                    currency=info.get("currency", "USD"),
                    asset_type="stock",
                ))
            except Exception as e:
                logger.warning(
                    "Failed to fetch growth info for %s: %s", ticker, e
                )

        return candidates

    def _scan_contrarian_value(
        self, config: dict[str, Any],
    ) -> list[Candidate]:
        """Scan contrarian value stocks matching filter criteria."""
        filters = config.get("filter", {})
        max_candidates = config.get("max_candidates", 10)

        # Curated seed list of large-cap value / dividend names.
        seed_tickers = [
            "VZ", "T", "IBM", "MMM", "DOW", "WBA", "KHC",
            "BMY", "GILD", "PFE", "WEC", "DUK", "SO", "NEE",
            "BHP", "RIO", "NEM", "FCX", "NUE",
            "PLD", "AMT", "SPG", "O", "VICI",
        ]

        pe_max = filters.get("pe_ratio_max", 999)
        yield_min = filters.get("dividend_yield_min", 0)
        pb_max = filters.get("price_to_book_max", 999)
        min_cap = filters.get("market_cap_min_usd", 0)
        allowed_sectors = {
            s.lower() for s in filters.get("sector_include", [])
        }

        candidates: list[Candidate] = []

        for ticker in seed_tickers:
            if len(candidates) >= max_candidates:
                break

            try:
                info = self._fetch_ticker_info(ticker)
                if info is None:
                    continue

                pe = float(info.get("trailingPE", 999) or 999)
                div_yield = float(info.get("dividendYield", 0) or 0)
                pb = float(info.get("priceToBook", 999) or 999)
                cap = float(info.get("marketCap", 0) or 0)
                sector = info.get("sector", "Unknown")

                if pe > pe_max:
                    continue
                if div_yield < yield_min:
                    continue
                if pb > pb_max:
                    continue
                if cap < min_cap:
                    continue
                if allowed_sectors and sector.lower() not in allowed_sectors:
                    continue

                candidates.append(Candidate(
                    ticker=ticker,
                    name=info.get("shortName", info.get("longName", ticker)),
                    exchange=info.get("exchange", "UNKNOWN"),
                    sector=sector,
                    market_cap_usd=cap,
                    source_universe="contrarian_value",
                    avg_daily_volume=float(
                        info.get("averageVolume", 0) or 0
                    ),
                    currency=info.get("currency", "USD"),
                    asset_type="stock",
                ))
            except Exception as e:
                logger.warning(
                    "Failed to fetch value info for %s: %s", ticker, e
                )

        return candidates

    def _fetch_ticker_info(self, ticker: str) -> dict[str, Any] | None:
        """Fetch basic info for a ticker via yfinance.

        Returns None if the fetch fails. Rate-limits requests.
        """
        try:
            import yfinance as yf
            time.sleep(0.3)  # Rate limiting.
            t = yf.Ticker(ticker)
            info = t.info
            if not info or info.get("regularMarketPrice") is None:
                return None
            return info
        except Exception as e:
            logger.debug("yfinance info fetch failed for %s: %s", ticker, e)
            return None

    def _apply_exclusions(
        self,
        candidates: list[Candidate],
        exclusion_set: set[str],
    ) -> list[Candidate]:
        """Remove candidates that are in the exclusion set."""
        return [
            c for c in candidates if c.ticker.upper() not in exclusion_set
        ]

    def _apply_liquidity_filter(
        self, candidates: list[Candidate],
    ) -> list[Candidate]:
        """Remove candidates below minimum average daily volume."""
        min_volume = self._scanner_config.get("min_avg_daily_volume", 0)
        if min_volume <= 0:
            return candidates
        return [c for c in candidates if c.avg_daily_volume >= min_volume]

    def _deduplicate(
        self, candidates: list[Candidate],
    ) -> list[Candidate]:
        """Remove duplicate tickers, keeping the first occurrence."""
        seen: set[str] = set()
        result: list[Candidate] = []
        for c in candidates:
            key = c.ticker.upper()
            if key not in seen:
                seen.add(key)
                result.append(c)
        return result


# ------------------------------------------------------------------
# Synthetic scanner for offline testing
# ------------------------------------------------------------------


class SyntheticScanner(AbstractScanner):
    """Returns fake candidates for offline testing."""

    def scan(self, exclusions: list[str] | None = None) -> list[Candidate]:
        """Return 15 synthetic candidates (5 per universe).

        Args:
            exclusions: Tickers to exclude.

        Returns:
            List of synthetic Candidate objects.
        """
        exclusion_set = {
            t.upper() for t in (exclusions or [])
        }

        candidates = [
            # US growth (5)
            Candidate("NVDA", "NVIDIA Corp", "NASDAQ", "Technology", 2.8e12, "us_growth", 45e6, "USD", "stock"),
            Candidate("MSFT", "Microsoft Corp", "NASDAQ", "Technology", 3.0e12, "us_growth", 25e6, "USD", "stock"),
            Candidate("GOOGL", "Alphabet Inc", "NASDAQ", "Communication Services", 2.0e12, "us_growth", 20e6, "USD", "stock"),
            Candidate("AMZN", "Amazon.com Inc", "NASDAQ", "Consumer Discretionary", 1.8e12, "us_growth", 30e6, "USD", "stock"),
            Candidate("META", "Meta Platforms Inc", "NASDAQ", "Communication Services", 1.2e12, "us_growth", 18e6, "USD", "stock"),
            # Sector ETFs (5)
            Candidate("SMH", "VanEck Semiconductor ETF", "NASDAQ", "Technology", 25e9, "sector_etfs", 8e6, "USD", "etf"),
            Candidate("IEMG", "iShares EM ETF", "NYSE", "Diversified", 70e9, "sector_etfs", 15e6, "USD", "etf"),
            Candidate("XLK", "Technology Select SPDR", "NYSE", "Technology", 60e9, "sector_etfs", 10e6, "USD", "etf"),
            Candidate("ICLN", "iShares Clean Energy ETF", "NASDAQ", "Utilities", 3e9, "sector_etfs", 2e6, "USD", "etf"),
            Candidate("EWJ", "iShares MSCI Japan ETF", "NYSE", "Diversified", 15e9, "sector_etfs", 5e6, "USD", "etf"),
            # Contrarian value (5)
            Candidate("RIO", "Rio Tinto plc", "NYSE", "Materials", 100e9, "contrarian_value", 3e6, "USD", "stock"),
            Candidate("NEE", "NextEra Energy", "NYSE", "Utilities", 150e9, "contrarian_value", 6e6, "USD", "stock"),
            Candidate("PLD", "Prologis Inc", "NYSE", "Real Estate", 110e9, "contrarian_value", 4e6, "USD", "stock"),
            Candidate("VZ", "Verizon Communications", "NYSE", "Communication Services", 170e9, "contrarian_value", 12e6, "USD", "stock"),
            Candidate("BMY", "Bristol-Myers Squibb", "NYSE", "Healthcare", 100e9, "contrarian_value", 8e6, "USD", "stock"),
        ]

        return [
            c for c in candidates if c.ticker.upper() not in exclusion_set
        ]
