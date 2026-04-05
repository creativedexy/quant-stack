"""Binance funding rate feed for perpetual futures proxy data.

Uses USDT perpetual contracts as a proxy for GBP spot pairs.
Negative funding rates indicate bearish perp sentiment, which
is a bullish signal for spot purchases.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.utils.exceptions import DataFeedError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_STALE_THRESHOLD_SECONDS = 8.5 * 3600  # 8.5 hours


class BinanceFundingFeed:
    """Fetches funding rate data from Binance perpetual futures.

    Args:
        client: An initialised python-binance Client with futures enabled.
    """

    def __init__(self, client: object) -> None:
        self._client = client

    def get_funding_rate(self, symbol: str) -> float:
        """Return the current funding rate as a decimal.

        For assets without a perpetual proxy (e.g. XRP), returns 0.0
        with a warning logged.

        Args:
            symbol: Spot trading pair (e.g. 'BTCGBP').

        Returns:
            Funding rate as decimal (e.g. -0.0001 = -0.01%).

        Raises:
            DataFeedError: If data is stale (timestamp > 8.5h ago).
        """
        from src.utils.config_loader import load_assets

        assets_cfg = load_assets()
        asset_cfg = assets_cfg.get("assets", {}).get(symbol, {})
        proxy = asset_cfg.get("funding_proxy")

        if proxy is None:
            logger.warning(
                "no_funding_proxy", symbol=symbol, message="No perpetual proxy, returning 0.0"
            )
            return 0.0

        try:
            rates = self._client.futures_funding_rate(symbol=proxy, limit=1)
        except Exception as exc:
            raise DataFeedError(f"Failed to fetch funding rate for {proxy}: {exc}") from exc

        if not rates:
            raise DataFeedError(f"No funding rate data returned for {proxy}")

        latest = rates[-1]
        rate = float(latest["fundingRate"])
        funding_time_ms = latest["fundingTime"]
        funding_dt = datetime.fromtimestamp(funding_time_ms / 1000, tz=timezone.utc)

        age_seconds = (datetime.now(timezone.utc) - funding_dt).total_seconds()
        if age_seconds > _STALE_THRESHOLD_SECONDS:
            raise DataFeedError(
                f"Funding rate data for {proxy} is stale: "
                f"{age_seconds / 3600:.1f}h old (threshold: 8.5h)"
            )

        return rate

    def get_funding_rate_8h_avg(self, symbol: str) -> float:
        """Return the average of the last 3 funding rate periods (24h average).

        Args:
            symbol: Spot trading pair (e.g. 'BTCGBP').

        Returns:
            Average funding rate as decimal.
        """
        from src.utils.config_loader import load_assets

        assets_cfg = load_assets()
        asset_cfg = assets_cfg.get("assets", {}).get(symbol, {})
        proxy = asset_cfg.get("funding_proxy")

        if proxy is None:
            return 0.0

        try:
            rates = self._client.futures_funding_rate(symbol=proxy, limit=3)
        except Exception as exc:
            raise DataFeedError(f"Failed to fetch funding rate history for {proxy}: {exc}") from exc

        if not rates:
            raise DataFeedError(f"No funding rate history for {proxy}")

        values = [float(r["fundingRate"]) for r in rates]
        return sum(values) / len(values)
