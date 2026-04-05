"""Offline backtest for smart-dca signal quality validation.

Replays historical hourly OHLCV data, computing SignalScorer composite
scores at each timestamp and simulating buy decisions.  Compares
signal-timed average entry prices against naive weekly DCA.

Usage:
    python -m scripts.backtest_signals --days 90 --assets BTCGBP,XRPGBP

Backtest simplifications vs live:
    * Sentiment signal uses a static Fear & Greed value of 50 (neutral).
    * Funding rate signal uses a mock value of 0.0 (neutral).
    * Buys are executed at the hour's close price (no slippage model).
    * No order sizing or budget tracking -- only entry timing is evaluated.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.signals.base import BaseSignal, SignalResult, clamp
from src.signals.scorer import SignalScorer
from src.signals.seasonality import BacktestSeasonalitySignal
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stub signals for backtest
# ---------------------------------------------------------------------------


class BacktestSentimentSignal(BaseSignal):
    """Static sentiment proxy for backtesting.

    Uses a fixed Fear & Greed value of 50 (neutral) because historical
    F&G data is not available offline.  This is a BACKTEST SIMPLIFICATION:
    live mode uses the real-time Alternative.me API.

    Args:
        config: Full settings dict from config_loader.
    """

    @property
    def name(self) -> str:
        return "sentiment"

    def compute(self, symbol: str, **kwargs: object) -> SignalResult:
        """Return a neutral sentiment score.

        Args:
            symbol: Trading pair (unused).
            **kwargs: Ignored.

        Returns:
            SignalResult with score=0.0 (neutral F&G of 50 is above
            the default fear threshold of 30, so no buy signal).
        """
        now: datetime = kwargs.get("now", datetime.now(timezone.utc))
        weight = self.config.get("scoring", {}).get(
            "signal_weights", {}
        ).get("sentiment", 0.20)
        # BACKTEST SIMPLIFICATION: static F&G index = 50 (neutral).
        # With default threshold of 30, index >= threshold -> score 0.
        return SignalResult(
            name=self.name,
            score=0.0,
            weight=weight,
            reason="Backtest: static F&G index 50 (neutral proxy)",
            timestamp=now,
        )


class BacktestFundingRateSignal(BaseSignal):
    """Static funding rate proxy for backtesting.

    Uses 0.0 (neutral) because historical per-hour funding rate data
    is not available offline.  This is a BACKTEST SIMPLIFICATION:
    live mode queries Binance perpetual futures.

    Args:
        config: Full settings dict from config_loader.
    """

    @property
    def name(self) -> str:
        return "funding_rate"

    def compute(self, symbol: str, **kwargs: object) -> SignalResult:
        """Return a neutral funding rate score.

        Args:
            symbol: Trading pair (unused).
            **kwargs: Ignored.

        Returns:
            SignalResult with score=0.0 (neutral funding).
        """
        now: datetime = kwargs.get("now", datetime.now(timezone.utc))
        weight = self.config.get("scoring", {}).get(
            "signal_weights", {}
        ).get("funding_rate", 0.25)
        # BACKTEST SIMPLIFICATION: mock funding rate = 0.0 (neutral).
        return SignalResult(
            name=self.name,
            score=0.0,
            weight=weight,
            reason="Backtest: mock funding rate 0.0 (neutral fallback)",
            timestamp=now,
        )


class BacktestMeanReversionSignal(BaseSignal):
    """Mean reversion signal that operates on a pre-loaded OHLCV DataFrame.

    Instead of calling the Binance API, this signal computes the 24h
    price change from the historical data available up to the current
    backtest timestamp.  Only data *before* the evaluation timestamp
    is used (no look-ahead).

    Args:
        config: Full settings dict from config_loader.
        ohlcv: Full OHLCV DataFrame for the backtest period.
    """

    def __init__(self, config: dict, ohlcv: pd.DataFrame) -> None:
        super().__init__(config)
        self._ohlcv = ohlcv

    @property
    def name(self) -> str:
        return "mean_reversion"

    def compute(self, symbol: str, **kwargs: object) -> SignalResult:
        """Compute mean reversion score from historical OHLCV.

        Args:
            symbol: Trading pair.
            **kwargs: Must include 'now' (datetime) for backtest time.

        Returns:
            SignalResult with dip-based score.
        """
        now: datetime = kwargs.get("now", datetime.now(timezone.utc))
        weight = self.config.get("scoring", {}).get(
            "signal_weights", {}
        ).get("mean_reversion", 0.35)

        signals_cfg = self.config.get("signals", {})
        dip_threshold = signals_cfg.get("mean_reversion_dip_pct", 4.0)
        window_hours = signals_cfg.get("mean_reversion_window_hours", 24)

        try:
            # NO LOOK-AHEAD: only use rows strictly before the current timestamp.
            past = self._ohlcv[self._ohlcv.index < now]
            if len(past) < 2:
                return SignalResult(
                    name=self.name,
                    score=0.0,
                    weight=weight,
                    reason="Insufficient historical data for mean reversion",
                    timestamp=now,
                )

            # Use the last `window_hours` rows available
            window = past.tail(min(window_hours, len(past)))
            first_close = float(window["close"].iloc[0])
            last_close = float(window["close"].iloc[-1])
            change_pct = ((last_close - first_close) / first_close) * 100.0

            if change_pct >= 0:
                return SignalResult(
                    name=self.name,
                    score=0.0,
                    weight=weight,
                    reason=(
                        f"{symbol} rose {change_pct:.1f}% in "
                        f"{len(window)}h window -> no mean reversion signal"
                    ),
                    timestamp=now,
                )

            score = clamp(abs(change_pct) / dip_threshold)
            return SignalResult(
                name=self.name,
                score=score,
                weight=weight,
                reason=(
                    f"{symbol} dropped {abs(change_pct):.1f}% in "
                    f"{len(window)}h window -> mean reversion signal"
                ),
                timestamp=now,
            )
        except Exception as exc:
            return SignalResult(
                name=self.name,
                score=0.0,
                weight=weight,
                reason=f"BacktestMeanReversionSignal failed: {exc}",
                timestamp=now,
            )


# ---------------------------------------------------------------------------
# Backtest data types
# ---------------------------------------------------------------------------


@dataclass
class BacktestBuy:
    """Record of a single simulated buy during backtest.

    Attributes:
        timestamp: UTC-aware datetime of the buy.
        price: Close price at the buy hour.
        composite_score: Weighted composite score at that moment.
    """

    timestamp: datetime
    price: float
    composite_score: float


@dataclass
class BacktestResult:
    """Aggregated metrics for a single asset backtest.

    Attributes:
        asset: Trading pair symbol.
        signal_avg_price: Average entry price from signal-timed buys.
        naive_avg_price: Average entry price from naive weekly DCA.
        improvement_pct: ((naive - signal) / naive) * 100.
        buy_count: Number of signal-triggered buys.
        weeks: Number of full weeks in the backtest period.
        buys_per_week: Average signal buys per week.
        worst_entry_vs_open: Highest buy price / that week's open.
        best_entry_vs_open: Lowest buy price / that week's open.
        buys: List of individual BacktestBuy records.
    """

    asset: str
    signal_avg_price: float
    naive_avg_price: float
    improvement_pct: float
    buy_count: int
    weeks: int
    buys_per_week: float
    worst_entry_vs_open: float
    best_entry_vs_open: float
    buys: list[BacktestBuy] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic OHLCV generation (offline)
# ---------------------------------------------------------------------------


def generate_synthetic_ohlcv(
    days: int, base_price: float = 50000.0, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic hourly OHLCV data for backtesting.

    Produces realistic-looking price data with random walks and
    occasional dips to validate signal behaviour.

    Args:
        days: Number of calendar days to generate.
        base_price: Starting close price.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
        Index is UTC-aware DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    n = days * 24

    # Random walk for close prices
    returns = rng.normal(0, 0.002, n)
    closes = base_price * np.cumprod(1 + returns)

    # Inject periodic dips (every ~7 days, a 3-5% dip)
    for week in range(days // 7):
        dip_idx = min(week * 168 + rng.integers(120, 168), n - 1)
        dip_pct = rng.uniform(0.03, 0.05)
        closes[dip_idx] = closes[dip_idx - 1] * (1 - dip_pct)

    opens = closes * (1 + rng.normal(0, 0.001, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volumes = rng.uniform(10, 200, n)

    timestamps = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        periods=n,
        freq="h",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=timestamps,
    )


# ---------------------------------------------------------------------------
# Naive DCA baseline
# ---------------------------------------------------------------------------


def compute_naive_dca_prices(ohlcv: pd.DataFrame) -> list[float]:
    """Extract Monday 09:00 UTC close prices as the naive DCA baseline.

    If a specific Monday 09:00 row does not exist in the data, the
    nearest available hour on that Monday is used.

    Args:
        ohlcv: Hourly OHLCV DataFrame with UTC DatetimeIndex.

    Returns:
        List of close prices at each weekly DCA point.
    """
    prices: list[float] = []
    start = ohlcv.index.min()
    end = ohlcv.index.max()

    # Find the first Monday at or after start
    current = start
    while current.weekday() != 0:
        current += timedelta(hours=1)
    # Snap to 09:00
    current = current.replace(hour=9, minute=0, second=0, microsecond=0)

    while current <= end:
        # Find closest available row
        mask = (ohlcv.index >= current - timedelta(hours=1)) & (
            ohlcv.index <= current + timedelta(hours=1)
        )
        candidates = ohlcv[mask]
        if not candidates.empty:
            prices.append(float(candidates["close"].iloc[0]))
        current += timedelta(weeks=1)

    return prices


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------


def run_backtest_for_asset(
    asset: str,
    ohlcv: pd.DataFrame,
    config: dict,
    buy_threshold: float | None = None,
) -> BacktestResult:
    """Replay OHLCV hour by hour, scoring and simulating buys.

    At each timestamp only data before that timestamp is available to
    the signals (no look-ahead).

    Args:
        asset: Trading pair symbol (e.g. 'BTCGBP').
        ohlcv: Hourly OHLCV DataFrame with UTC DatetimeIndex.
        config: Full settings dict.
        buy_threshold: Override for scoring.buy_threshold.

    Returns:
        BacktestResult with all computed metrics.
    """
    # Apply buy_threshold override if supplied
    effective_config = config
    if buy_threshold is not None:
        effective_config = {**config}
        effective_config["scoring"] = {
            **config.get("scoring", {}),
            "buy_threshold": buy_threshold,
        }

    # Build backtest signals
    # Start with a dummy time; we update it each iteration
    first_ts = ohlcv.index[0].to_pydatetime()
    seasonality = BacktestSeasonalitySignal(effective_config, first_ts)
    mean_rev = BacktestMeanReversionSignal(effective_config, ohlcv)
    funding = BacktestFundingRateSignal(effective_config)
    sentiment = BacktestSentimentSignal(effective_config)

    signals: list[BaseSignal] = [seasonality, mean_rev, funding, sentiment]
    scorer = SignalScorer(signals, effective_config)

    buys: list[BacktestBuy] = []

    for ts in ohlcv.index:
        ts_dt = ts.to_pydatetime()

        # Update the seasonality signal's backtest time
        seasonality.backtest_time = ts_dt

        # Compute composite score -- pass 'now' for signals that accept it
        # The scorer calls signal.compute(symbol, **feed_kwargs)
        # BacktestMeanReversionSignal and stub signals accept 'now'
        composite = scorer.compute(asset, now=ts_dt)

        if composite.should_buy:
            close_price = float(ohlcv.loc[ts, "close"])
            buys.append(
                BacktestBuy(
                    timestamp=ts_dt,
                    price=close_price,
                    composite_score=composite.score,
                )
            )

    # Naive DCA prices
    naive_prices = compute_naive_dca_prices(ohlcv)

    # Compute metrics
    signal_avg = np.mean([b.price for b in buys]) if buys else 0.0
    naive_avg = np.mean(naive_prices) if naive_prices else 0.0

    if naive_avg > 0:
        improvement = ((naive_avg - signal_avg) / naive_avg) * 100.0
    else:
        improvement = 0.0

    total_hours = len(ohlcv)
    weeks = max(total_hours / (7 * 24), 1.0)
    buys_per_week = len(buys) / weeks

    # Worst/best entry relative to that week's open
    worst_ratio = 0.0
    best_ratio = float("inf") if buys else 0.0

    for buy in buys:
        # Find the Monday 00:00 of the buy's ISO week
        buy_dt = buy.timestamp
        week_start = buy_dt - timedelta(days=buy_dt.weekday())
        week_start = week_start.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        # Get close price nearest to week start
        mask = (ohlcv.index >= week_start) & (
            ohlcv.index < week_start + timedelta(hours=1)
        )
        week_rows = ohlcv[mask]
        if not week_rows.empty:
            week_open = float(week_rows["close"].iloc[0])
            if week_open > 0:
                ratio = buy.price / week_open
                worst_ratio = max(worst_ratio, ratio)
                best_ratio = min(best_ratio, ratio)

    if best_ratio == float("inf"):
        best_ratio = 0.0

    return BacktestResult(
        asset=asset,
        signal_avg_price=float(signal_avg),
        naive_avg_price=float(naive_avg),
        improvement_pct=float(improvement),
        buy_count=len(buys),
        weeks=int(weeks),
        buys_per_week=float(buys_per_week),
        worst_entry_vs_open=float(worst_ratio),
        best_entry_vs_open=float(best_ratio),
        buys=buys,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_results(results: list[BacktestResult]) -> None:
    """Print a summary table of backtest results to stdout.

    Args:
        results: List of BacktestResult objects.
    """
    header = (
        f"{'Asset':<12} {'Signal Avg':>12} {'Naive Avg':>12} "
        f"{'Improv %':>10} {'Buys':>6} {'Buys/Wk':>8} "
        f"{'Worst':>8} {'Best':>8}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print("SMART-DCA SIGNAL BACKTEST RESULTS")
    print(separator)
    print(header)
    print(separator)

    for r in results:
        print(
            f"{r.asset:<12} {r.signal_avg_price:>12.2f} "
            f"{r.naive_avg_price:>12.2f} {r.improvement_pct:>10.2f} "
            f"{r.buy_count:>6} {r.buys_per_week:>8.2f} "
            f"{r.worst_entry_vs_open:>8.4f} {r.best_entry_vs_open:>8.4f}"
        )

    print(separator)
    print(
        "NOTE: Sentiment uses static F&G=50 (neutral proxy). "
        "Funding rate uses mock 0.0. These are approximate."
    )
    print()


def save_results_csv(
    results: list[BacktestResult], path: Path
) -> None:
    """Save backtest results to a CSV file.

    Args:
        results: List of BacktestResult objects.
        path: Output CSV file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "asset",
            "signal_avg_price",
            "naive_avg_price",
            "improvement_pct",
            "buy_count",
            "weeks",
            "buys_per_week",
            "worst_entry_vs_open",
            "best_entry_vs_open",
        ])
        for r in results:
            writer.writerow([
                r.asset,
                f"{r.signal_avg_price:.4f}",
                f"{r.naive_avg_price:.4f}",
                f"{r.improvement_pct:.4f}",
                r.buy_count,
                r.weeks,
                f"{r.buys_per_week:.4f}",
                f"{r.worst_entry_vs_open:.4f}",
                f"{r.best_entry_vs_open:.4f}",
            ])

    logger.info("backtest_csv_saved", path=str(path), assets=len(results))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the backtest CLI.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Backtest smart-dca signal quality against historical OHLCV data.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Backtest period in calendar days (default: 90).",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default="",
        help="Comma-separated asset symbols (default: all enabled in config).",
    )
    return parser


def main() -> None:
    """Entry point for the backtest CLI."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    # Load config -- fall back to defaults if config files are unavailable
    try:
        from src.utils.config_loader import load_assets, load_settings
        config = load_settings()
        assets_config = load_assets()
    except Exception:
        logger.warning(
            "config_load_failed",
            message="Using default settings for backtest",
        )
        config = {
            "scoring": {
                "buy_threshold": 65,
                "signal_weights": {
                    "seasonality": 0.20,
                    "mean_reversion": 0.35,
                    "funding_rate": 0.25,
                    "sentiment": 0.20,
                },
            },
            "signals": {
                "mean_reversion_dip_pct": 4.0,
                "mean_reversion_window_hours": 24,
            },
        }
        assets_config = {
            "assets": {
                "BTCGBP": {"enabled": True},
                "ETHGBP": {"enabled": False},
            }
        }

    # Determine which assets to backtest
    if args.assets:
        asset_symbols = [s.strip() for s in args.assets.split(",")]
    else:
        asset_symbols = [
            sym
            for sym, cfg in assets_config.get("assets", {}).items()
            if cfg.get("enabled", False)
        ]

    if not asset_symbols:
        print("No enabled assets found. Use --assets to specify symbols.")
        sys.exit(1)

    logger.info(
        "backtest_starting",
        days=args.days,
        assets=asset_symbols,
    )
    print(
        f"Generating {args.days} days of synthetic OHLCV data "
        f"for {len(asset_symbols)} asset(s)..."
    )
    print(
        "NOTE: Sentiment signal uses static F&G=50 (approximate). "
        "Funding rate uses mock 0.0."
    )

    results: list[BacktestResult] = []
    for asset in asset_symbols:
        # Each asset gets its own synthetic series (different seed)
        seed = hash(asset) % (2**31)
        ohlcv = generate_synthetic_ohlcv(args.days, seed=seed)
        result = run_backtest_for_asset(asset, ohlcv, config)
        results.append(result)

    print_results(results)

    csv_path = Path("data") / "backtest_results.csv"
    save_results_csv(results, csv_path)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
