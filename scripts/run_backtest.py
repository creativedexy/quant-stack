#!/usr/bin/env python3
"""Run a backtest for a named strategy on synthetic or processed data.

Usage:
    python -m scripts.run_backtest --strategy mean_reversion
    python -m scripts.run_backtest --strategy momentum --start 2018-01-01 --end 2023-12-31
    python -m scripts.run_backtest --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestEngine
from src.backtest.strategy import strategy_registry
from src.data.synthetic import generate_synthetic_ohlcv
from src.features.pipeline import FeaturePipeline
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    available = strategy_registry.list_strategies()

    parser = argparse.ArgumentParser(description="Run a backtest")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=available,
        help="Strategy name to backtest",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all registered strategies and print comparison",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: from config)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: all available data)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker to backtest (default: first in universe)",
    )

    args = parser.parse_args()

    if not args.compare and args.strategy is None:
        parser.error("Either --strategy or --compare is required")

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

    # Optionally truncate to end date
    if args.end:
        ohlcv = ohlcv.loc[:args.end]

    # Build features (compat mode — OHLCV + indicators)
    logger.info("Building features...")
    pipeline = FeaturePipeline()
    features = pipeline.run(ohlcv)

    # Backtest engine
    engine = BacktestEngine(config)

    # Ensure reports directory exists
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    if args.compare:
        # Run all registered strategies
        strategies = [strategy_registry.create(name) for name in available]
        comparison = engine.compare(strategies, features, features)
        _print_comparison(comparison)

        # Save individual plots
        for name in available:
            strat = strategy_registry.create(name)
            result = engine.run(strat, features, features)
            save_path = reports_dir / f"backtest_{name}.png"
            engine.plot_results(result, save_path=save_path)
            logger.info("Saved plot for %s to %s", name, save_path)

    else:
        strat = strategy_registry.create(args.strategy)
        result = engine.run(strat, features, features)
        _print_result(result)

        # Save equity curve plot
        save_path = reports_dir / f"backtest_{args.strategy}.png"
        engine.plot_results(result, save_path=save_path)
        logger.info("Saved plot to %s", save_path)


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
    fmt = comparison.copy()
    for col in fmt.columns:
        if fmt[col].dtype == float:
            fmt[col] = fmt[col].map(lambda x: f"{x:.4f}")
    print(fmt.to_string())
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
