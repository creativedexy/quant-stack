#!/usr/bin/env python3
"""Run a backtest for a named strategy on synthetic or processed data.

Usage:
    python -m scripts.run_backtest --strategy mean_reversion
    python -m scripts.run_backtest --strategy momentum --start 2018-01-01
    python -m scripts.run_backtest --strategy mean_reversion --strategy momentum
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import run_backtest, compare_strategies
from src.backtest.strategy import get_strategy, STRATEGY_REGISTRY
from src.data.synthetic import generate_synthetic_ohlcv
from src.features.pipeline import FeaturePipeline
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest")
    parser.add_argument(
        "--strategy",
        type=str,
        nargs="+",
        required=True,
        choices=sorted(STRATEGY_REGISTRY),
        help="Strategy name(s) to backtest",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: from config)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker to backtest (default: first in universe)",
    )

    args = parser.parse_args()
    config = load_config()

    start = args.start or config["data"]["start_date"]
    ticker = args.ticker or config["universe"]["tickers"][0]

    # Generate or load data
    logger.info("Generating synthetic data for %s from %s", ticker, start)
    ohlcv = generate_synthetic_ohlcv(
        ticker=ticker,
        days=1260,
        start_date=start,
        seed=config["general"].get("random_seed", 42),
    )

    # Build features
    logger.info("Building features…")
    pipeline = FeaturePipeline()
    features = pipeline.run(ohlcv)

    # Run backtest(s)
    strategies = [get_strategy(name, config) for name in args.strategy]

    if len(strategies) == 1:
        result = run_backtest(strategies[0], features, config)
        _print_result(result)
    else:
        comparison = compare_strategies(strategies, features, config)
        _print_comparison(comparison)


def _print_result(result) -> None:
    """Print a single backtest result to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  Strategy: {result.strategy_name}")
    print(f"{'=' * 60}")
    for key, val in result.metrics.items():
        if isinstance(val, float):
            print(f"  {key:.<30s} {val:>12.4f}")
        else:
            print(f"  {key:.<30s} {val!s:>12}")
    print(f"{'=' * 60}\n")


def _print_comparison(comparison) -> None:
    """Print a multi-strategy comparison table."""
    print(f"\n{'=' * 80}")
    print("  Strategy Comparison")
    print(f"{'=' * 80}")
    # Format for readability
    fmt = comparison.copy()
    for col in fmt.columns:
        if fmt[col].dtype == float:
            fmt[col] = fmt[col].map(lambda x: f"{x:.4f}")
    print(fmt.to_string())
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
