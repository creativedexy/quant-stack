#!/usr/bin/env python3
"""Generate and optionally execute a portfolio rebalance.

In **paper mode** (the default) the script prints the trade plan.
With ``--execute`` it sends orders to the paper broker.

In **live mode** (``--execute --live``) two gates are required:
both flags AND a typed confirmation prompt.

Usage:
    python -m scripts.run_rebalance                     # dry run, paper
    python -m scripts.run_rebalance --execute            # paper, executes
    python -m scripts.run_rebalance --execute --live     # LIVE, confirmation

CRITICAL: ``--live`` without ``--execute`` is an error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data.synthetic import generate_multi_asset_data
from src.execution.broker import create_broker
from src.execution.oms import OrderManagementSystem
from src.portfolio.optimiser import PortfolioOptimiser
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run portfolio rebalance")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Optimisation method (default: from config)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the trade plan (default: dry run only)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable LIVE execution (requires --execute and confirmation)",
    )

    args = parser.parse_args()

    # CRITICAL: --live without --execute is an error
    if args.live and not args.execute:
        print(
            "ERROR: --live requires --execute. "
            "Both flags are needed for live trading.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = load_config()

    # Override mode if --live flag is set
    if args.live:
        config.setdefault("execution", {})["mode"] = "live"

    mode = config.get("execution", {}).get("mode", "paper")
    tickers = config["universe"]["tickers"]

    # ---------------------------------------------------------------
    # 1. Generate synthetic data (stand-in for loading processed data)
    # ---------------------------------------------------------------
    logger.info("Loading data for %d tickers", len(tickers))
    multi_data = generate_multi_asset_data(
        tickers,
        days=504,
        seed=config["general"].get("random_seed", 42),
    )

    # Build a returns DataFrame from close prices
    closes = pd.DataFrame({t: df["Close"] for t, df in multi_data.items()})
    returns = closes.pct_change().dropna()

    # ---------------------------------------------------------------
    # 2. Optimise portfolio
    # ---------------------------------------------------------------
    optimiser = PortfolioOptimiser(config=config)
    method = args.method or config.get("portfolio", {}).get("default_method")
    target_weights = optimiser.optimise(returns, method=method)

    print(f"\nTarget weights ({method}):")
    for ticker, w in target_weights.items():
        print(f"  {ticker}: {w:.4f}")
    print()

    # ---------------------------------------------------------------
    # 3. Create broker and get current state
    # ---------------------------------------------------------------
    broker = create_broker(config)
    broker.connect()

    current_positions = broker.get_positions()
    account_value = broker.get_account_value()
    if account_value <= 0:
        # Fallback for brokers that don't report value
        account_value = float(
            config.get("backtest", {}).get("initial_capital", 100_000)
        )

    # Use last close as current prices
    prices = {t: float(df["Close"].iloc[-1]) for t, df in multi_data.items()}

    # ---------------------------------------------------------------
    # 4. Compute rebalance orders
    # ---------------------------------------------------------------
    oms = OrderManagementSystem(broker, config=config)
    orders = oms.compute_rebalance_orders(
        target_weights, current_positions, account_value, prices,
    )

    if not orders:
        print("No trades required — portfolio is already at target.")
        broker.disconnect()
        return

    # ---------------------------------------------------------------
    # 5. Print order plan
    # ---------------------------------------------------------------
    print(f"{'=' * 70}")
    print(f"  Rebalance Plan ({len(orders)} orders, mode={mode})")
    print(f"{'=' * 70}")
    for order in orders:
        print(
            f"  {order.side.upper():4s} {order.quantity:>8g} "
            f"{order.ticker:<12s} {order.reason}"
        )
    print(f"{'=' * 70}\n")

    # ---------------------------------------------------------------
    # 6. Execute (with safety gates)
    # ---------------------------------------------------------------
    if not args.execute:
        print("[DRY RUN] No orders sent. Use --execute to trade.\n")
        oms.execute_plan(orders, dry_run=True)
        broker.disconnect()
        return

    if mode == "live":
        # ---- LIVE CONFIRMATION GATE ----
        print("WARNING: You are about to execute LIVE orders.")
        print(f"  Orders: {len(orders)}")
        print(f"  Portfolio value: {account_value:,.0f}")
        confirm = input("TYPE 'CONFIRM LIVE TRADING' TO PROCEED: ").strip()
        if confirm != "CONFIRM LIVE TRADING":
            print("Aborted — confirmation did not match.")
            broker.disconnect()
            sys.exit(1)

    # Execute
    if mode != "live":
        print("[PAPER MODE] Executing simulated orders...")
    else:
        print("Executing LIVE orders...")

    report = oms.execute_plan(orders, dry_run=False)
    print(
        f"Done: {len(report.orders_executed)} executed, "
        f"{len(report.orders_failed)} failed "
        f"out of {len(report.orders_planned)} planned\n"
    )

    # ---------------------------------------------------------------
    # 7. Reconcile
    # ---------------------------------------------------------------
    recon = oms.reconcile(target_weights, account_value, prices)
    print("Reconciliation:")
    if recon["aligned"]:
        print("  All positions within tolerance.")
    else:
        print(f"  Total drift: {recon['total_drift']:.4f}")
        for disc in recon["discrepancies"]:
            print(
                f"  {disc['ticker']}: target={disc['target_weight']:.4f}, "
                f"actual={disc['actual_weight']:.4f}, "
                f"diff={disc['diff']:+.4f}"
            )
    print()

    broker.disconnect()


if __name__ == "__main__":
    main()
