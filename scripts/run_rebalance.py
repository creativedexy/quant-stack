#!/usr/bin/env python3
"""Generate and optionally execute a portfolio rebalance.

In **paper mode** (the default) the script prints the trade plan and logs
what *would* be executed, without touching any broker.

In **live mode** (``--live`` flag) two explicit confirmations are required
before any order is sent.

Usage:
    python -m scripts.run_rebalance
    python -m scripts.run_rebalance --method equal_weight
    python -m scripts.run_rebalance --live   # requires double confirmation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic import generate_multi_asset_data
from src.execution.broker import IBBroker
from src.execution.oms import OrderManager
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
        "--live",
        action="store_true",
        help="Enable LIVE execution (requires double confirmation)",
    )

    args = parser.parse_args()
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
    import pandas as pd
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
    # 3. Generate trade plan
    # ---------------------------------------------------------------
    broker = IBBroker(config=config)
    broker.connect()

    current_positions = broker.get_positions()
    account = broker.get_account_summary()
    portfolio_value = account.get("net_liquidation", 100_000.0)

    # Use last close as current prices
    prices = {t: float(df["Close"].iloc[-1]) for t, df in multi_data.items()}

    oms = OrderManager(config=config)
    plan = oms.generate_plan(target_weights, current_positions, portfolio_value, prices)

    if not plan:
        print("No trades required — portfolio is already at target.")
        broker.disconnect()
        return

    # ---------------------------------------------------------------
    # 4. Print plan
    # ---------------------------------------------------------------
    print(f"{'=' * 60}")
    print(f"  Rebalance Plan ({len(plan)} orders, mode={mode})")
    print(f"{'=' * 60}")
    for order in plan:
        print(f"  {order.side:4s} {order.quantity:>6d} {order.ticker:<12s} {order.reason}")
    print(f"{'=' * 60}\n")

    # ---------------------------------------------------------------
    # 5. Execute (with safety gates)
    # ---------------------------------------------------------------
    if mode != "live":
        print("[PAPER MODE] Executing simulated orders…")
        result = oms.execute_plan(plan, broker)
        print(f"Done: {result.orders_sent} sent, {result.orders_filled} filled (paper)\n")
    else:
        # ---- DOUBLE CONFIRMATION GATE ----
        print("WARNING: You are about to execute LIVE orders.")
        print(f"  Orders: {len(plan)}")
        print(f"  Portfolio value: £{portfolio_value:,.0f}")
        confirm1 = input("Type 'CONFIRM' to proceed: ").strip()
        if confirm1 != "CONFIRM":
            print("Aborted — first confirmation failed.")
            broker.disconnect()
            sys.exit(1)

        confirm2 = input("Type 'EXECUTE LIVE TRADES' to send orders: ").strip()
        if confirm2 != "EXECUTE LIVE TRADES":
            print("Aborted — second confirmation failed.")
            broker.disconnect()
            sys.exit(1)

        print("Executing LIVE orders…")
        result = oms.execute_plan(plan, broker)
        print(f"Done: {result.orders_sent} sent, {result.orders_filled} filled\n")

        # Reconcile
        actual = broker.get_positions()
        report = oms.reconcile(target_weights, actual, prices, portfolio_value)
        print("Reconciliation report:")
        print(report.to_string())
        print()

    broker.disconnect()


if __name__ == "__main__":
    main()
